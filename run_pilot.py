#!/usr/bin/env python3
"""ECPM Phase-2 live-exploration pilot harness (runs INSIDE the repo,
pinned to the freeze).

Two pilots: the matched seed-7 pair (deterministic + stochastic). For each
pilot, the model actively explores the MDP itself -- picking its own
actions step by step, first on the pre-change world (M0, "learn the
network"), then (after a reset) on the post-change world (M1, "does it
notice and stop using the link that just broke") -- and only afterwards
answers the 4 frozen probes (detection / localization / preservation /
adaptation) as a continuation of its own exploration transcript. See
explore_agent.py for the loop itself; this file only wires it to a
provider (with retry/backoff) and writes the artifact.

This replaces the earlier handed-over-evidence pilot (which asked the same
4 probes about a transcript a non-LLM simulator collected, rather than one
the model gathered itself) -- that pipeline is gone; resource_mdp.py's
underlying evidence-collection machinery (paired_evidence, prompt_view,
etc.) is untouched and stays FROZEN per INTERFACE.md, it's just no longer
what this harness runs.

Each artifact retains: the full exploration transcript and per-episode
step logs, the resulting metrics, the 4 probes' raw responses / parsed /
scored results, seeds/model settings, and the env freeze SHA the run is
pinned to.

Usage (from the repo root, branch v2.1-prefreeze):

  python3 run_pilot.py                                   # dry-run, no API
  ANTHROPIC_API_KEY=... python3 run_pilot.py \
      --provider anthropic --model claude-sonnet-4-6
  OPENAI_API_KEY=... python3 run_pilot.py \
      --provider openai --model gpt-4o --base-url https://api.openai.com/v1
  AZURE_OPENAI_API_KEY=... python3 run_pilot.py \
      --provider azure --model YOUR-DEPLOYMENT \
      --azure-endpoint https://YOUR-RESOURCE.openai.azure.com

Outputs: pilot_deterministic.json / pilot_stochastic.json (or *_dryrun.json)
in --out (default: pilot_artifacts/). stdlib only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict

import explore_agent
import explore_metrics
from ecpm_parser import run_probe
from resource_mdp import CONDITIONS, SCHEMA_VERSION, make_pair, pair_to_json

FROZEN_SHA = "5318c3e113438c563c5676d58252d84fda22aa49"
GRAPH_SEED = 7
PROBES = ("detection", "localization", "preservation", "adaptation")

ASKS = {
    "detection": (
        'Question: across your two rounds of exploring this network (the '
        'first set of episodes, then the reset and second set), did the '
        'network\'s dynamics change at any point?\nAnswer with exactly '
        'one JSON object: {"changed": true} or {"changed": false}. No '
        'other text.'),
    "localization": (
        'The dynamics changed at some point during your exploration. '
        'Question: which single (node, action) pair changed?\nAnswer '
        'with exactly one JSON object: {"node": "<node>", "action": '
        '"<aK>"}. No other text.'),
    "preservation": (
        'For EACH of the following (node, action) pairs, judge whether '
        'its dynamics changed at any point during your exploration:\n'
        '{queried}\nAnswer with exactly one JSON object of the form '
        '{{"pairs": [{{"node": "...", "action": "...", "changed": '
        'true|false}}, ...]}} containing every listed pair exactly once. '
        'No other text.'),
    "adaptation": (
        'Plan a route for the CURRENT network (as of your most recent '
        'exploration) from {start} to {goal}. Answer with exactly one '
        'JSON object of the form {{"route": [{{"node": "...", "action": '
        '"..."}}, ...]}}: at most 32 steps, the first step\'s node must '
        'be {start}, each next step\'s node must be where the previous '
        'action leads, and the route must end at {goal}. No other '
        'text.'),
}


def git_head():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def queried_pairs_for(record, n_other=3):
    """Deterministic probe set: the target pair + n_other unchanged pairs,
    order shuffled by a fixed seed."""
    ch = record["change"]
    target = (ch["edge"]["from"], ch["action"])
    everyone = sorted((node, a) for node, menu in
                      record["legal_actions_pre"].items() for a in menu)
    others = [p for p in everyone if p != target]
    rng = random.Random(f"pilot|{GRAPH_SEED}|preservation")
    picked = rng.sample(others, n_other) + [target]
    rng.shuffle(picked)
    return [{"node": n, "action": a} for n, a in picked]


# ---------------------------------------------------------------- providers


class TransientLLMError(Exception):
    """Empty/unparseable LLM response body -- treated as retryable by
    with_retry, same spirit as a network error."""


def with_retry(fn, *args, max_attempts=6, base_delay=1.0, max_delay=30.0,
               **kwargs):
    """Call fn(*args, **kwargs), retrying transient failures with jittered
    exponential backoff (honoring a Retry-After header when present).
    Retryable: HTTP 429/500/502/503/504, network/timeout errors, and
    empty/unparseable response bodies (TransientLLMError, KeyError,
    json.JSONDecodeError). Everything else (4xx auth/bad-request errors)
    raises immediately. Stdlib only -- reimplements the retry pattern seen
    in reference material, no third-party dependency added, matching this
    repo's stdlib-only convention."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except urllib.error.HTTPError as ex:
            if ex.code not in (429, 500, 502, 503, 504):
                raise
            last_exc = ex
            retry_after = ex.headers.get("Retry-After") if ex.headers else None
            delay = (float(retry_after) if retry_after
                     else min(max_delay, base_delay * 2 ** (attempt - 1)))
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                json.JSONDecodeError, KeyError, TransientLLMError) as ex:
            last_exc = ex
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
        if attempt == max_attempts:
            raise last_exc
        delay *= random.uniform(0.5, 1.5)
        print(f"retryable error (attempt {attempt}/{max_attempts}), "
              f"retrying in {delay:.1f}s: {last_exc}")
        time.sleep(delay)


def call_anthropic_chat(model, system, messages, max_tokens):
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": model, "max_tokens": max_tokens,
                         "temperature": 0, "system": system,
                         "messages": messages}).encode(),
        headers={"content-type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = "".join(b.get("text", "") for b in data.get("content", []))
    if not text.strip():
        raise TransientLLMError("empty Anthropic response content")
    return text, data.get("usage", {})


def call_openai_chat(model, system, messages, max_tokens, base_url):
    full_messages = [{"role": "system", "content": system}] + list(messages)
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "max_tokens": max_tokens,
                         "temperature": 0,
                         "messages": full_messages}).encode(),
        headers={"content-type": "application/json",
                 "authorization":
                     f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    if not text.strip():
        raise TransientLLMError("empty OpenAI response content")
    return text, data.get("usage", {})


def call_azure_chat(deployment, system, messages, max_tokens, endpoint,
                    api_version):
    full_messages = [{"role": "system", "content": system}] + list(messages)
    url = (endpoint.rstrip("/") + "/openai/deployments/" + deployment
           + "/chat/completions?api-version=" + api_version)
    req = urllib.request.Request(
        url,
        data=json.dumps({"max_tokens": max_tokens, "temperature": 0,
                         "messages": full_messages}).encode(),
        headers={"content-type": "application/json",
                 "api-key": os.environ["AZURE_OPENAI_API_KEY"]})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    if not text.strip():
        raise TransientLLMError("empty Azure response content")
    return text, data.get("usage", {})


def dry_run_answer(record, probe, queried):
    """Oracle-derived canned replies (wrapped in prose/fences to exercise
    the extractor). Pipeline demo only; provider is labeled 'dry-run'."""
    ch = record["change"]
    if probe == "detection":
        return 'Looking back: ```json\n{"changed": true}\n```'
    if probe == "localization":
        obj = {"node": ch["edge"]["from"], "action": ch["action"]}
        return "My answer: " + json.dumps(obj)
    if probe == "preservation":
        target = (ch["edge"]["from"], ch["action"])
        pairs = [{"node": q["node"], "action": q["action"],
                  "changed": (q["node"], q["action"]) == target}
                 for q in queried]
        return json.dumps({"pairs": pairs})
    o = record["oracle"]["post"]
    steps = [{"node": n, "action": a}
             for n, a in zip(o["optimal_route"], o["optimal_actions"])]
    return json.dumps({"route": steps})


# ------------------------------------------------------------------- pilot


def _episode_to_json(ep):
    return asdict(ep)


def run_pilot(deterministic, args):
    """Live-exploration pilot: the model picks its own actions on M0, then
    (after a reset) on M1, before answering the existing 4 frozen probes
    on a fork of its own exploration transcript. See explore_agent.py for
    the loop itself; this function only wires it to a provider and writes
    the artifact."""
    inst = make_pair(GRAPH_SEED, args.condition,
                     deterministic=deterministic, matched=True)
    record = json.loads(json.dumps(pair_to_json(inst)))
    queried = queried_pairs_for(record)
    cfg = explore_agent.ExploreConfig(
        max_episodes_m0=args.m0_episodes, max_episodes_m1=args.m1_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        announce_change=args.announce_change,
        max_context_tokens_est=args.explore_context_budget,
        seed=GRAPH_SEED)

    last_usage = {}

    def act_fn(system, messages):
        nonlocal last_usage
        if args.provider == "anthropic":
            text, usage = with_retry(call_anthropic_chat, args.model,
                                     system, messages, args.max_tokens)
        elif args.provider == "azure":
            text, usage = with_retry(call_azure_chat, args.model, system,
                                     messages, args.max_tokens,
                                     args.azure_endpoint, args.api_version)
        elif args.provider == "openai":
            text, usage = with_retry(call_openai_chat, args.model, system,
                                     messages, args.max_tokens,
                                     args.base_url)
        else:
            raise AssertionError("dry-run must not call act_fn")
        last_usage = usage
        return text

    if args.provider == "dry-run":
        result = explore_agent.run_explore_instance(
            inst, cfg, node_policy_fn=lambda mdp: explore_agent.dry_run_policy(
                mdp, inst.labels))
    else:
        result = explore_agent.run_explore_instance(inst, cfg, act_fn=act_fn)

    metrics = explore_metrics.compute_explore_metrics(
        inst, result["m0_episodes"], result["m1_episodes"])
    head = git_head()
    artifact = {
        "pilot": "deterministic" if deterministic else "stochastic",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "env": {"schema_version": SCHEMA_VERSION,
                "frozen_sha": FROZEN_SHA,
                "git_head": head,
                "pinned_to_freeze": head == FROZEN_SHA},
        "instance": {"graph_seed": GRAPH_SEED, "condition": args.condition,
                     "deterministic": deterministic, "matched": True,
                     "seeds": record["seeds"]},
        "model": {"provider": args.provider, "model": args.model,
                  "temperature": 0, "max_tokens": args.max_tokens},
        "explore": {
            "config": asdict(cfg),
            "transcript": result["messages"],
            "m0_episodes": [_episode_to_json(e)
                            for e in result["m0_episodes"]],
            "m1_episodes": [_episode_to_json(e)
                            for e in result["m1_episodes"]],
            "metrics": metrics,
        },
        "probes": {},
    }

    system_prompt = explore_agent.build_system_prompt(record["goal"])
    for probe in PROBES:
        ask = ASKS[probe]
        if probe == "preservation":
            listed = "\n".join(f'- node {q["node"]}, action {q["action"]}'
                               for q in queried)
            ask = ask.format(queried=listed)
        elif probe == "adaptation":
            ask = ask.format(start=record["start"], goal=record["goal"])
        forked_messages = list(result["messages"]) + [
            {"role": "user", "content": ask}]

        if args.provider == "dry-run":
            raw, usage = dry_run_answer(record, probe, queried), {}
        else:
            raw = act_fn(system_prompt, forked_messages)
            usage = last_usage

        probe_result = run_probe(record, probe, raw,
                                 queried_pairs=(queried
                                                if probe == "preservation"
                                                else None))
        artifact["probes"][probe] = {
            "prompt_chars": len(ask),
            "prompt_tokens_est": len(ask) // 4,
            "queried_pairs": queried if probe == "preservation" else None,
            "raw_response": raw,
            "provider_usage": usage,
            "parsed": probe_result["parsed"],
            "scored": probe_result["scored"],
        }
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="dry-run",
                    choices=["dry-run", "anthropic", "openai", "azure"])
    ap.add_argument("--azure-endpoint",
                    default="https://YOUR-RESOURCE.openai.azure.com")
    ap.add_argument("--api-version", default="2024-06-01")
    ap.add_argument("--model", default="dry-run")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--mode", default="both",
                    choices=["both", "det", "sto"])
    ap.add_argument("--out", default="pilot_artifacts")
    ap.add_argument("--condition", default="silent_break",
                    choices=list(CONDITIONS))
    ap.add_argument("--m0-episodes", type=int, default=4)
    ap.add_argument("--m1-episodes", type=int, default=4)
    ap.add_argument("--max-steps-per-episode", type=int, default=25)
    ap.add_argument("--announce-change", action="store_true",
                    help="explicitly tell the model reliabilities may "
                         "have changed at the M0->M1 reset (ablation; "
                         "default is silent, matching the PI's "
                         "must-infer-from-observation criterion)")
    ap.add_argument("--explore-context-budget", type=int, default=12000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    todo = {"both": (True, False), "det": (True,), "sto": (False,)}[args.mode]
    for det in todo:
        art = run_pilot(det, args)
        suffix = "_dryrun" if args.provider == "dry-run" else ""
        name = ("pilot_deterministic" if det else "pilot_stochastic")
        path = os.path.join(args.out, f"{name}{suffix}.json")
        with open(path, "w") as fh:
            json.dump(art, fh, indent=2)
        summary = {p: art["probes"][p]["scored"].get("status",
                   art["probes"][p]["scored"].get("correct"))
                   for p in PROBES}
        print(f"{path}: pinned={art['env']['pinned_to_freeze']} "
              f"statuses={summary}")


if __name__ == "__main__":
    main()
