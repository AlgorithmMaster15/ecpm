"""References' Localization restricted to the queried pairs under
--queried-target-half. Stdlib only, no API calls.

Stochastic silent break, K = 10, over the eligible run set (run_set.sto) of
the seed file. The queried set is run_pilot.queried_pairs_for_icl with
target_half=True: the target is among the five on seeds where
run_pilot.queried_target_included(seed) is True, otherwise all five are
unchanged links. Each reference localizes by argmax of its per-pair score
(reference_sweep.pair_scores) over the queried pairs only, ties broken by
pair name (reference_sweep.ranked). On a seed without the target queried
the restricted localization is a miss whenever the reference names any
queried pair.

Usage:
    python3 experiments/queried_restricted.py \\
        --seeds-file runs/seed_eligibility_31_130.json \\
        --json runs/references/queried_restricted.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_pilot as P
from experiments import reference_sweep as RS
from experiments.baseline_k_sweep import true_pair

NAMES = ("rate", "transition", "bayes", "generator")


def seed_row(seed, k=10):
    got = RS.instance(seed, "silent_break", False, k)
    if got is None:
        return None
    _, record, pv = got
    rec = json.loads(json.dumps(record))
    queried = {(q["node"], q["action"]) for q in P.queried_pairs_for_icl(
        rec, {"seed": seed, "matched": True}, target_half=True)}
    truth = true_pair(rec)
    scores = RS.pair_scores(pv, False)
    row = {"seed": seed, "target_included_coin": P.queried_target_included(seed),
           "target_queried": truth in queried, "loc": {}, "hit": {}}
    for name in NAMES:
        sc = {p: v for p, v in scores[name].items() if p in queried}
        top = RS.ranked(sc)[0] if sc else None
        row["loc"][name] = list(top) if top else None
        row["hit"][name] = top is not None and top == truth
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", default="runs/seed_eligibility_31_130.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    seeds = json.load(open(a.seeds_file))["run_set"]["sto"]
    rows = [r for r in (seed_row(s, a.k) for s in seeds) if r]
    q = [r for r in rows if r["target_queried"]]
    out = {
        "meta": {"seeds_file": a.seeds_file, "run_set": "sto",
                 "condition": "silent_break", "mode": "sto", "k": a.k,
                 "selection": "run_pilot.queried_pairs_for_icl(target_half=True)",
                 "localize": "argmax of reference_sweep.pair_scores over queried pairs"},
        "n": len(rows),
        "n_target_queried": len(q),
        "share_target_queried": round(len(q) / len(rows), 4) if rows else None,
        "localization_given_target_queried": {
            n: round(sum(r["hit"][n] for r in q) / len(q), 4) if q else None
            for n in NAMES},
        "localization_over_all_seeds": {
            n: round(sum(r["hit"][n] for r in rows) / len(rows), 4) if rows else None
            for n in NAMES},
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
