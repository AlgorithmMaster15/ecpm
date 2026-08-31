#!/usr/bin/env python3
"""ECPM Phase-2 pilot harness (runs INSIDE the repo, pinned to the freeze).

Two pilots: the matched seed-7 pair (deterministic + stochastic,
silent_break). For each pilot, all four probes are run end to end:

  prompt_view (prompt-safe payload) -> prompt text -> model ->
  raw response -> frozen parser -> scoring -> artifact JSON

Each artifact retains exactly the agreed list: prompt-safe payload,
raw model response, parser and execution statuses, route/regret outputs,
seeds/model settings, and realized event/token counts, plus the env
freeze SHA the run is pinned to.

A second pilot type, active exploration (--pilot-type active), runs the
model as a live agent instead: it picks its own actions step by step
against the real MDP (see explore_agent.py), first on the pre-change
world (M0), then after a reset on the post-change world (M1), and only
afterwards answers the same 4 probes as a continuation of its own
exploration transcript, instead of being handed a pre-collected
evidence log. This is an addition alongside the original pilot, not a
replacement: resource_mdp.py's evidence-collection machinery
(paired_evidence, prompt_view) stays untouched and FROZEN either way,
--pilot-type passive (the default) still uses it exactly as before.

Usage (from the repo root, branch v2.1-prefreeze):

  python3 run_pilot.py                                   # dry-run, no API
  ANTHROPIC_API_KEY=... python3 run_pilot.py \
      --provider anthropic --model claude-sonnet-4-6
  OPENAI_API_KEY=... python3 run_pilot.py \
      --provider openai --model gpt-4o --base-url https://api.openai.com/v1
  AZURE_OPENAI_API_KEY=... python3 run_pilot.py \
      --provider azure --model YOUR-DEPLOYMENT \
      --azure-endpoint https://YOUR-RESOURCE.openai.azure.com
  python3 run_pilot.py --pilot-type active                # active exploration instead

Outputs: pilot_deterministic.json / pilot_stochastic.json (or *_dryrun.json,
or *_active[_dryrun].json for --pilot-type active) in --out (default:
pilot_artifacts/). stdlib only.
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
from resource_mdp import (CONDITIONS, SCHEMA_VERSION, make_pair, pair_to_json,
                          paired_evidence, prompt_view)

FROZEN_SHA = "5318c3e113438c563c5676d58252d84fda22aa49"
GRAPH_SEED = 7
PROBES = ("detection", "localization", "preservation", "adaptation")

INTRO = """You are analysing a courier network. Nodes are locations; at each
node you may attempt the listed actions (aK). An attempt either delivers
you to that action's destination or you stay and retry (each attempt
costs 1). You observed the network in two periods.

Nodes: {nodes}
Start: {start}   Goal: {goal}

Action menu, period A (earlier): {menu_pre}
Action menu, period B (later): {menu_post}

Observations, period A:
{ev_pre}

Observations, period B:
{ev_post}
"""

ASKS = {
    "detection": (
        'Question: did the network\'s dynamics change between period A and '
        'period B?\nAnswer with exactly one JSON object: '
        '{"changed": true} or {"changed": false}. No other text.'),
    "localization": (
        'The dynamics changed between the periods. Question: which single '
        '(node, action) pair changed?\nAnswer with exactly one JSON '
        'object: {"node": "<node>", "action": "<aK>"}. No other text.'),
    "preservation": (
        'For EACH of the following (node, action) pairs, judge whether its '
        'dynamics changed between period A and period B:\n{queried}\n'
        'Answer with exactly one JSON object of the form '
        '{{"pairs": [{{"node": "...", "action": "...", "changed": '
        'true|false}}, ...]}} containing every listed pair exactly once. '
        'No other text.'),
    "adaptation": (
        'Plan a route for period B (the later network) from {start} to '
        '{goal}. Answer with exactly one JSON object of the form '
        '{{"route": [{{"node": "...", "action": "..."}}, ...]}}: at most '
        '32 steps, the first step\'s node must be {start}, each next '
        'step\'s node must be where the previous action leads, and the '
        'route must end at {goal}. No other text.'),
}

# Same 4 probes, worded for the active-exploration pilot: the model refers
# to its own exploration episodes instead of a handed-over period A/B log.
ASKS_ACTIVE = {
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
        'Plan a route for the current network (as of your most recent '
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


def build_record(deterministic):
    inst = make_pair(GRAPH_SEED, "silent_break",
                     deterministic=deterministic, matched=True)
    ev = paired_evidence(inst, k=5, evidence_seed=0)
    return json.loads(json.dumps(pair_to_json(inst, ev)))


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


def build_prompt(record, view, probe, queried):
    menus = {p: "; ".join(f"{node}: {', '.join(m)}" for node, m in
                          sorted(view[f"legal_actions_{p}"].items()))
             for p in ("pre", "post")}
    intro = INTRO.format(nodes=", ".join(view["nodes"]),
                         start=view["start"], goal=view["goal"],
                         menu_pre=menus["pre"], menu_post=menus["post"],
                         ev_pre=view["evidence"]["pre"],
                         ev_post=view["evidence"]["post"])
    ask = ASKS[probe]
    if probe == "preservation":
        listed = "\n".join(f'- node {q["node"]}, action {q["action"]}'
                           for q in queried)
        ask = ask.format(queried=listed)
    elif probe == "adaptation":
        ask = ask.format(start=view["start"], goal=view["goal"])
    return intro + "\n" + ask + "\n"


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


def call_anthropic(model, prompt, max_tokens):
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": model, "max_tokens": max_tokens,
                         "temperature": 0,
                         "messages": [{"role": "user",
                                       "content": prompt}]}).encode(),
        headers={"content-type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = "".join(b.get("text", "") for b in data.get("content", []))
    return text, data.get("usage", {})


def call_azure(deployment, prompt, max_tokens, endpoint, api_version):
    url = (endpoint.rstrip("/") + "/openai/deployments/" + deployment
           + "/chat/completions?api-version=" + api_version)
    req = urllib.request.Request(
        url,
        data=json.dumps({"max_tokens": max_tokens, "temperature": 0,
                         "messages": [{"role": "user",
                                       "content": prompt}]}).encode(),
        headers={"content-type": "application/json",
                 "api-key": os.environ["AZURE_OPENAI_API_KEY"]})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    return text, data.get("usage", {})


def call_openai(model, prompt, max_tokens, base_url):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "max_tokens": max_tokens,
                         "temperature": 0,
                         "messages": [{"role": "user",
                                       "content": prompt}]}).encode(),
        headers={"content-type": "application/json",
                 "authorization":
                     f"Bearer {os.environ['OPENAI_API_KEY']}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    return text, data.get("usage", {})


# Multi-turn variants for the active-exploration pilot (system + a growing
# message list, matching explore_agent.py's act_fn(system, messages)
# contract) instead of a single one-shot prompt. Wrapped in with_retry,
# unlike the single-shot passive-mode callers above.


def call_anthropic_chat(model, system, messages, max_tokens, thinking_budget=0):
    """Calls Claude with the given system prompt and message history.
    If thinking_budget > 0, enables Extended Thinking with that token
    budget (Anthropic requires temperature 1 and max_tokens greater than
    thinking_budget in that case) and returns the thinking content
    separately from the visible answer."""
    body = {"model": model, "max_tokens": max_tokens, "system": system,
            "messages": messages}
    if thinking_budget > 0:
        body["thinking"] = {"type": "enabled",
                            "budget_tokens": thinking_budget}
        body["temperature"] = 1
    else:
        body["temperature"] = 0
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    reasoning = "".join(b.get("thinking", "") for b in data.get("content", [])
                        if b.get("type") == "thinking")
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    if not text.strip():
        raise TransientLLMError("empty Anthropic response content")
    return text, reasoning, data.get("usage", {})


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
        return 'Looking at period B: ```json\n{"changed": true}\n```'
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


def run_pilot(deterministic, args):
    """Passive pilot: a non-LLM simulator collects the evidence log up
    front and hands it to the model as text. See run_pilot_active below
    for the live-agent counterpart (--pilot-type active)."""
    record = build_record(deterministic)
    view = prompt_view(record, rendering=args.rendering,
                       periods=("pre", "post"),
                       budget_per_pair=args.budget)
    queried = queried_pairs_for(record)
    head = git_head()
    artifact = {
        "pilot": "deterministic" if deterministic else "stochastic",
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "env": {"schema_version": SCHEMA_VERSION,
                "frozen_sha": FROZEN_SHA,
                "git_head": head,
                "pinned_to_freeze": head == FROZEN_SHA},
        "instance": {"graph_seed": GRAPH_SEED, "condition": "silent_break",
                     "deterministic": deterministic, "matched": True,
                     "k_per_pair": record["evidence"]["k_per_pair"],
                     "evidence_seed": record["evidence"]["evidence_seed"],
                     "seeds": record["seeds"]},
        "model": {"provider": args.provider, "model": args.model,
                  "temperature": 0, "max_tokens": args.max_tokens,
                  "rendering": args.rendering,
                  "budget_per_pair": args.budget},
        "prompt_safe_payload": view,
        "probes": {},
    }
    for probe in PROBES:
        prompt = build_prompt(record, view, probe, queried)
        if args.provider == "anthropic":
            raw, usage = call_anthropic(args.model, prompt, args.max_tokens)
        elif args.provider == "azure":
            raw, usage = call_azure(args.model, prompt, args.max_tokens,
                                    args.azure_endpoint, args.api_version)
        elif args.provider == "openai":
            raw, usage = call_openai(args.model, prompt, args.max_tokens,
                                     args.base_url)
        else:
            raw, usage = dry_run_answer(record, probe, queried), {}
        result = run_probe(record, probe, raw,
                           queried_pairs=(queried if probe == "preservation"
                                          else None))
        artifact["probes"][probe] = {
            "prompt_chars": len(prompt),
            "prompt_tokens_est": len(prompt) // 4,
            "queried_pairs": queried if probe == "preservation" else None,
            "prompt_text": prompt,
            "raw_response": raw,
            "provider_usage": usage,
            "parsed": result["parsed"],
            "scored": result["scored"],
        }
    return artifact


def _episode_to_json(ep):
    return asdict(ep)


def run_pilot_active(deterministic, args):
    """Active-exploration pilot: the model picks its own actions on M0,
    then (after a reset) on M1, before answering the same 4 frozen probes
    on a fork of its own exploration transcript, instead of being
    handed a pre-collected evidence log (see run_pilot above). See
    explore_agent.py for the loop itself; this function only wires it to
    a provider and writes the artifact."""
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

    # Passed into explore_agent for each exploration turn, and reused below for the probes.
    def act_fn(system, messages):
        nonlocal last_usage  # so the caller can read usage after the call, since only (text, reasoning) is returned
        reasoning = ""
        if args.provider == "anthropic":
            text, reasoning, usage = with_retry(
                call_anthropic_chat, args.model, system, messages,
                args.max_tokens, args.thinking_budget)
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
        return text, reasoning

    if args.provider == "dry-run":
        # no API calls: a scripted policy stands in for the model's actions
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
        ask = ASKS_ACTIVE[probe]
        if probe == "preservation":
            listed = "\n".join(f'- node {q["node"]}, action {q["action"]}'
                               for q in queried)
            ask = ask.format(queried=listed)
        elif probe == "adaptation":
            ask = ask.format(start=record["start"], goal=record["goal"])
        # each probe appends to a copy of the exploration transcript, not a fresh one
        forked_messages = list(result["messages"]) + [
            {"role": "user", "content": ask}]

        if args.provider == "dry-run":
            raw, reasoning, usage = dry_run_answer(record, probe, queried), "", {}
        else:
            raw, reasoning = act_fn(system_prompt, forked_messages)
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
            "reasoning": reasoning,
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
    ap.add_argument("--rendering", default="F2_shuffled",
                    help="passive pilot-type only")
    ap.add_argument("--budget", type=int, default=5,
                    help="passive pilot-type only")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--mode", default="both",
                    choices=["both", "det", "sto"])
    ap.add_argument("--out", default="pilot_artifacts")
    ap.add_argument("--pilot-type", default="passive",
                    choices=["passive", "active"],
                    help="passive (default): hand the model a "
                         "pre-collected evidence log, as before. active: "
                         "let the model explore the MDP itself, picking "
                         "its own actions (see explore_agent.py).")
    ap.add_argument("--condition", default="silent_break",
                    choices=list(CONDITIONS), help="active pilot-type only")
    ap.add_argument("--m0-episodes", type=int, default=4,
                    help="active pilot-type only")
    ap.add_argument("--m1-episodes", type=int, default=4,
                    help="active pilot-type only")
    ap.add_argument("--max-steps-per-episode", type=int, default=25,
                    help="active pilot-type only")
    ap.add_argument("--announce-change", action="store_true",
                    help="active pilot-type only: explicitly tell the "
                         "model reliabilities may have changed at the "
                         "M0->M1 reset (ablation; default is silent, "
                         "requiring the model to infer the change from "
                         "observation alone)")
    ap.add_argument("--explore-context-budget", type=int, default=12000,
                    help="active pilot-type only")
    ap.add_argument("--thinking-budget", type=int, default=0,
                    help="active pilot-type only, Anthropic provider "
                         "only: greater than 0 enables Claude Extended "
                         "Thinking with this token budget (requires "
                         "--max-tokens greater than this value)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    todo = {"both": (True, False), "det": (True,), "sto": (False,)}[args.mode]
    for det in todo:
        if args.pilot_type == "active":
            art = run_pilot_active(det, args)
        else:
            art = run_pilot(det, args)
        suffix = "_dryrun" if args.provider == "dry-run" else ""
        if args.pilot_type == "active":
            suffix = "_active" + suffix
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
