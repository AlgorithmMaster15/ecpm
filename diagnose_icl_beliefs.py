#!/usr/bin/env python3
"""Why did a level get its Period B beliefs wrong?

For every queried pair in every Turn B answer, this prints what the model
reported next to three reference values: Period A truth, Period B truth,
and the Period B empirical rate in the visible logs. The pattern across
those columns is the diagnosis.

    ANCHORED  reported value matches Period A truth, not Period B
    EVIDENCE  reported value matches what the logs actually showed
    TRUTH     reported value matches Period B truth

A level scoring badly because it copied the supplied Period A graph looks
different from one scoring badly because it read misleading evidence
correctly, and the aggregate cannot tell them apart.

    python3 diagnose_icl_beliefs.py pilot_artifacts/sol_degr_sto_b5
    python3 diagnose_icl_beliefs.py pilot_artifacts/... --level graph_a
"""

import argparse
import collections
import glob
import json
import os

import run_pilot

TOLERANCE = 0.05


def truth_map(record, period):
    return {(edge["from"], edge["action"]): (edge["to"], edge["p"])
            for edge in record[f"world_{period}"]["edges"]}


def empirical(view, period):
    attempts, successes = collections.Counter(), collections.Counter()
    for row in run_pilot.raw_visible_rows(view, period):
        node, action, nxt = [part.strip() for part in row.strip("[]").split(",")]
        attempts[(node, action)] += 1
        if nxt != node:
            successes[(node, action)] += 1
    return {key: successes[key] / attempts[key] for key in attempts}


def rebuild(artifact):
    scenario = dict(run_pilot.SCENARIO_DEFAULTS)
    scenario.update(artifact["scenario"])
    deterministic = artifact["instance"]["deterministic"]
    record = run_pilot.build_record(scenario, deterministic)
    view = run_pilot.prompt_view(record, rendering="F2_shuffled",
                                 periods=("pre", "post"),
                                 budget_per_pair=scenario["budget"],
                                 budget_seed=0)
    return record, view


def close(a, b):
    return a is not None and b is not None and abs(a - b) <= TOLERANCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--level", default=None, help="restrict to one level")
    args = ap.parse_args()

    paths = [p for p in sorted(glob.glob(os.path.join(args.root, "*.json")))
             if os.path.basename(p) != "summary.json"]
    tally = collections.Counter()
    cache = {}

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("protocol") != "icl_two_response_v1":
            continue
        if args.level and artifact["level"] != args.level:
            continue
        key = (artifact["scenario"]["seed"], artifact["scenario"]["budget"],
               artifact["scenario"]["condition"],
               artifact["instance"]["deterministic"])
        if key not in cache:
            cache[key] = rebuild(artifact)
        record, view = cache[key]
        pre, post = truth_map(record, "pre"), truth_map(record, "post")
        seen = empirical(view, "post")

        beliefs = artifact["turns"]["B"]["parsed"]["beliefs"]
        print(f"\n{artifact['level']}  repeat {artifact['repeat']}  "
              f"budget {artifact['scenario']['budget']}")
        print(f"  {'pair':7s} {'said':>6s} {'A_true':>7s} {'B_true':>7s} "
              f"{'B_seen':>7s}   verdict")
        for pair in beliefs.get("pairs", []):
            k = (pair["node"], pair["action"])
            said = pair.get("p_success")
            a_true = pre.get(k, (None, None))[1]
            b_true = post.get(k, (None, None))[1]
            b_seen = seen.get(k)
            labels = []
            if close(said, b_true):
                labels.append("TRUTH")
            if close(said, a_true) and not close(a_true, b_true):
                labels.append("ANCHORED")
            if close(said, b_seen) and not close(said, b_true):
                labels.append("EVIDENCE")
            verdict = "+".join(labels) if labels else "other"
            tally[(artifact["level"], verdict)] += 1
            fmt = lambda v: "  null" if v is None else f"{v:6.2f}"
            print(f"  {k[0]}:{k[1]:4s} {fmt(said)} {fmt(a_true)} "
                  f"{fmt(b_true)} {fmt(b_seen)}   {verdict}")

    print("\ntally")
    for (level, verdict), count in sorted(tally.items()):
        print(f"  {level:10s} {verdict:20s} {count}")


if __name__ == "__main__":
    main()
