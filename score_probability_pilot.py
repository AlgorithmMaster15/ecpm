#!/usr/bin/env python3
"""Scorer for the exploratory probability-elicitation pilot (seed-7 pair).

Runs INSIDE the repo root on the frozen commit (5318c3e), on top of the
signed-off tree; nothing in v2.1-prefreeze, Section 7, or schema 2.1 is
touched. The route half of the response is scored by the FROZEN grader
(ecpm_parser.run_probe); only the probability half uses the small logic
below.

Usage, from the repo root:

    python3 score_probability_pilot.py --mode sto response.json
    python3 score_probability_pilot.py --mode det response.json --out report.json

Expected response format (see PACKETS_README.md):

    {"estimates": [{"node": "A", "action": "a1", "p_pre": 0.8, "p_post": 0.8},
                   ... every listed pair exactly once ...],
     "route": [{"node": "E", "action": "a2"}, ...]}
"""

import argparse
import json
import re
import subprocess
import sys

from ecpm_parser import run_probe
from resource_mdp import make_pair, pair_to_json, paired_evidence, prompt_view

FROZEN = "5318c3e113438c563c5676d58252d84fda22aa49"
SEED = 7


def build(mode):
    inst = make_pair(SEED, "silent_break",
                     deterministic=(mode == "det"), matched=True)
    ev = paired_evidence(inst, k=5, evidence_seed=0)
    rec = json.loads(json.dumps(pair_to_json(inst, ev)))
    view = prompt_view(rec, rendering="F2_shuffled",
                       periods=("pre", "post"), budget_per_pair=5)
    return rec, view


def true_p(rec, period):
    return {(e["from"], e["action"]): e["p"]
            for e in rec["world_" + period]["edges"]}


def visible_rates(view, period):
    """Success rates computable from the rendered prompt evidence."""
    counts = {}
    for n, a, dest in re.findall(r"\((\w+), (a\d+), (\w+)\)",
                                 view["evidence"][period]):
        s, t = counts.get((n, a), (0, 0))
        counts[(n, a)] = (s + (dest != n), t + 1)
    return {k: (s / t if t else None) for k, (s, t) in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["det", "sto"])
    ap.add_argument("response", help="model response JSON file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    if head != FROZEN:
        print("WARNING: HEAD is not the frozen commit; results are not "
              "comparable (" + head[:7] + ")")

    rec, view = build(args.mode)
    pairs = sorted((n, a) for n, menu in rec["legal_actions_pre"].items()
                   for a in menu)
    tp = {"pre": true_p(rec, "pre"), "post": true_p(rec, "post")}
    vr = {"pre": visible_rates(view, "pre"),
          "post": visible_rates(view, "post")}
    resp = json.load(open(args.response))

    # ---- probability half: coverage, then error ------------------------
    report = {"mode": args.mode, "frozen": head == FROZEN}
    est = {}
    problems = []
    for e in resp.get("estimates", []):
        key = (e.get("node"), e.get("action"))
        if key not in set(pairs):
            problems.append("unknown pair " + str(key))
            continue
        if key in est:
            problems.append("duplicate pair " + str(key))
            continue
        try:
            pp, po = float(e["p_pre"]), float(e["p_post"])
        except (KeyError, TypeError, ValueError):
            problems.append("bad probabilities for " + str(key))
            continue
        if not (0 <= pp <= 1 and 0 <= po <= 1):
            problems.append("out of range for " + str(key))
            continue
        est[key] = (pp, po)
    missing = [p for p in pairs if p not in est]
    report["coverage"] = {"listed": len(pairs), "scored": len(est),
                          "missing": [f"{n} {a}" for n, a in missing],
                          "problems": problems}

    if est:
        for period, idx in (("pre", 0), ("post", 1)):
            report["mae_vs_true_" + period] = round(
                sum(abs(est[k][idx] - tp[period][k]) for k in est) / len(est), 4)
            report["mae_vs_visible_" + period] = round(
                sum(abs(est[k][idx] - vr[period][k]) for k in est) / len(est), 4)
        ch = (rec["change"]["edge"]["from"], rec["change"]["action"])
        report["changed_pair"] = {
            "pair": f"{ch[0]} {ch[1]}",
            "true": [tp["pre"][ch], tp["post"][ch]],
            "visible": [vr["pre"][ch], vr["post"][ch]],
            "model": list(est[ch]) if ch in est else None,
        }
        gaps = {k: abs(v[1] - v[0]) for k, v in est.items()}
        top = max(gaps.values())
        tied = sorted(k for k, g in gaps.items() if abs(g - top) < 1e-9)
        report["elicited_localization"] = {
            "argmax_gap_pairs": [f"{n} {a}" for n, a in tied],
            "gap": round(top, 4),
            "matches_true_change": ch in tied,
            "tie": len(tied) > 1,
        }

    # ---- route half: frozen grader ------------------------------------
    route = resp.get("route")
    if route is not None:
        result = run_probe(rec, "adaptation", json.dumps({"route": route}))
        report["route"] = result["scored"]

    print(json.dumps(report, indent=2))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
