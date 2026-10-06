"""Evidence-only references on the paper's seeds. Stdlib only, no API calls.

Four references answer from the same prompt view a model sees:
  rate        the repo's rate counter (ecpm_baseline.run_baseline)
  transition  total-variation distance between the two periods' outcome
              distributions per pair
  bayes       log Bayes factor, separate vs shared Dirichlet-multinomial
              outcome distributions (uniform prior, alpha = 1)
  generator   posterior over no change and each pair under the generator's
              prior (p ~ U[0.60, 0.95] on a 400-point grid, zero / halve /
              redirect, every pair equally likely). Rebuilt from the paper's
              Appendix A description: values that do not match the paper are
              reported as mismatches, never forced.
All four route identically (ecpm_baseline.plan_route on Period B evidence);
the reference route is scored with resource_mdp.score_route.

Usage:
    python3 experiments/reference_sweep.py --seeds 31-130 --json runs/references/references.json
    python3 experiments/reference_sweep.py --seeds 1000-1099 --json runs/references/heldout.json
"""
import argparse, json, math, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecpm_baseline as B
import resource_mdp as R
from experiments.baseline_k_sweep import parse_seeds, true_pair

CONDITIONS = ("no_change", "irrelevant", "silent_break", "hard_removal", "redirect", "degradation")
GRID = [0.60 + 0.35 * (i + 0.5) / 400 for i in range(400)]


def outcomes(events):
    """Per pair: Counter of outcomes, where a drop is 'drop' and a delivery is its destination."""
    out = {}
    for node, action, dest in events:
        d = out.setdefault((node, action), {})
        key = "drop" if dest == node else dest
        d[key] = d.get(key, 0) + 1
    return out


def periods(pv):
    ev = pv["evidence"]
    pre = outcomes(B.parse_events(ev["pre"]))
    post = outcomes(B.parse_events(ev["post"]))
    return pre, post


def tv(a, b):
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


def log_dm(counts, keys):
    n = sum(counts.get(k, 0) for k in keys)
    A = len(keys)
    return math.lgamma(A) - math.lgamma(n + A) + sum(math.lgamma(counts.get(k, 0) + 1) for k in keys)


def log_bf(a, b):
    keys = sorted(set(a) | set(b))
    both = {k: a.get(k, 0) + b.get(k, 0) for k in keys}
    return log_dm(a, keys) + log_dm(b, keys) - log_dm(both, keys)


def _lse(xs):
    m = max(xs)
    return m if m == -math.inf else m + math.log(sum(math.exp(x - m) for x in xs))


def gen_loglik(a, b, grid):
    """Log likelihood of one pair's (pre, post) outcomes under unchanged / zero / halve / redirect."""
    s0 = sum(v for k, v in a.items() if k != "drop"); f0 = a.get("drop", 0)
    s1 = sum(v for k, v in b.items() if k != "drop"); f1 = b.get("drop", 0)
    d0 = {k for k in a if k != "drop"}; d1 = {k for k in b if k != "drop"}
    same_dest = not d0 or not d1 or (d1 <= d0 and len(d0 | d1) == 1)
    moved_dest = bool(d0) and bool(d1) and not (d0 & d1) and len(d1) == 1

    def lp(p, s, f):
        if p <= 0:
            return 0.0 if s == 0 else -math.inf
        if p >= 1:
            return 0.0 if f == 0 else -math.inf
        return s * math.log(p) + f * math.log(1 - p)
    G = math.log(len(grid))
    unch = _lse([lp(p, s0 + s1, f0 + f1) for p in grid]) - G if same_dest else -math.inf
    zero = (_lse([lp(p, s0, f0) for p in grid]) - G) if s1 == 0 else -math.inf
    halve = (_lse([lp(p, s0, f0) + lp(max(0.05, round(p * 0.5, 2)), s1, f1) for p in grid]) - G) if same_dest else -math.inf
    redir = (_lse([lp(p, s0 + s1, f0 + f1) for p in grid]) - G - math.log(6)) if moved_dest else -math.inf
    return unch, {"zero": zero, "halve": halve, "redirect": redir}


PRIOR = {"zero": 2 / 6, "halve": 1 / 6, "redirect": 1 / 6}  # irrelevant + silent break both zero p


def ranked(scores):
    """Pairs sorted by score descending, ties broken by pair name (as ecpm_baseline does)."""
    return sorted(scores, key=lambda k: (-scores[k], k))


def references(pv, deterministic):
    pre, post = periods(pv)
    common = [k for k in pre if k in post]
    removed = [tuple(p) for p in B.menu_removals(pv)]
    out = {}
    tvs = {k: tv(pre[k], post[k]) for k in common}
    bfs = {k: log_bf(pre[k], post[k]) for k in common}
    grid = [1.0] if deterministic else GRID
    post_ratio = {}
    for k in common:
        u, ch = gen_loglik(pre[k], post[k], grid)
        for t, l in ch.items():
            post_ratio[(k, t)] = math.log(PRIOR[t]) + l - (u if u > -math.inf else -1e9)
    for name, sc in (("transition", tvs), ("bayes", bfs)):
        if removed:
            out[name] = {"loc": removed[0], "score": math.inf}
        elif sc:
            top = ranked(sc)[0]
            out[name] = {"loc": top, "score": sc[top]}
        else:
            out[name] = None
    if removed:
        out["generator"] = {"loc": removed[0], "score": math.inf}
    elif post_ratio:
        best = max(post_ratio, key=lambda kt: (post_ratio[kt], [-ord(c) for c in "".join(kt[0])]))
        # P(no change) vs P(best change): log odds; prior of no change is 1/6 spread over all pairs.
        n = len(common)
        lo = _lse([v - math.log(n) for v in post_ratio.values()]) - math.log(1 / 6)
        out["generator"] = {"loc": best[0], "score": lo}
    else:
        out["generator"] = None
    return out


def pair_scores(pv, deterministic):
    """Additive: every pair's score under each reference, {name: {pair: score}}.

    rate is |rate_post - rate_pre| (ecpm_baseline.rank_pairs), transition
    and bayes as in references(), generator the best change-type log
    posterior ratio per pair. A pair removed from the menu scores inf under
    every reference, as references() localizes it outright. Higher means
    more likely the changed pair; argmax with ranked() tie-breaking."""
    pre, post = periods(pv)
    common = [k for k in pre if k in post]
    grid = [1.0] if deterministic else GRID
    out = {"rate": {tuple(x["pair"]): x["abs_delta"]
                    for x in B.rank_pairs(*B.read_periods(pv)[:2])},
           "transition": {k: tv(pre[k], post[k]) for k in common},
           "bayes": {k: log_bf(pre[k], post[k]) for k in common},
           "generator": {}}
    for k in common:
        u, ch = gen_loglik(pre[k], post[k], grid)
        vals = [math.log(PRIOR[t]) + l - (u if u > -math.inf else -1e9)
                for t, l in ch.items()]
        if vals:
            out["generator"][k] = max(vals)
    for pair in B.menu_removals(pv):
        for name in out:
            out[name][tuple(pair)] = math.inf
    return out


def instance(seed, condition, deterministic, k, rendering="F2_shuffled", n_nodes=None):
    try:
        kw = {} if n_nodes is None else {"n_nodes": n_nodes}
        inst = R.make_pair(seed, condition, deterministic=deterministic, matched=True, **kw)
        ev = R.paired_evidence(inst, k=k, max_episodes=300 * max(1, k // 5), horizon=60 * max(1, k // 5))
        record = R.pair_to_json(inst, ev)
    except (ValueError, AssertionError, KeyError, RuntimeError, TypeError):
        return None
    return inst, record, R.prompt_view(record, rendering=rendering, budget_per_pair=k)


def row(seed, condition, deterministic, k, threshold):
    got = instance(seed, condition, deterministic, k)
    if got is None:
        return None
    inst, record, pv = got
    t0 = time.process_time()
    res = B.run_baseline(pv, delta_threshold=threshold)
    t_rate = time.process_time() - t0
    t0 = time.process_time()
    refs = references(pv, deterministic)
    t_refs = time.process_time() - t0
    truth = true_pair(record)
    _, post_stats, _, _ = B.read_periods(pv)
    top = B.rank_pairs(*B.read_periods(pv)[:2])
    route = (res.get("adaptation") or {}).get("route") or []
    path = R.route_from_actions(inst.start, [s["action"] for s in route], inst.labels, inst.m1) if route else None
    sc = R.score_route(inst.m1, path, inst.start)
    r = {"seed": seed, "condition": condition, "mode": "det" if deterministic else "sto", "k": k,
         "truth": list(truth) if truth else None,
         "rate": {"loc": list(res["localization"]) if res.get("localization") else None,
                  "detected": bool(res.get("detection")),
                  "score": (math.inf if res.get("localization_basis") == "menu_removal"
                            else (top[0]["abs_delta"] if top else 0.0))},
         "route_optimal": bool(sc.get("status") == "valid_finite" and abs(sc.get("regret", 1)) < 1e-9),
         "cpu_s": {"rate": t_rate, "others": t_refs},
         "prompt_bytes": len(json.dumps(pv["evidence"]).encode())}
    for name in ("transition", "bayes", "generator"):
        v = refs.get(name)
        r[name] = {"loc": list(v["loc"]) if v else None, "score": v["score"] if v else None}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="31-130")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--modes", nargs="+", default=["sto", "det"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    rows, t = [], time.time()
    for mode in a.modes:
        for k in a.k:
            thr = B.CALIBRATED_DELTA_THRESHOLD.get(k)
            for cond in a.conditions:
                if mode == "det" and cond == "degradation":
                    continue
                for s in parse_seeds(a.seeds):
                    r = row(s, cond, mode == "det", k, thr)
                    if r:
                        rows.append(r)
    for r in rows:  # json cannot hold inf
        for name in ("rate", "transition", "bayes", "generator"):
            if r[name] and r[name]["score"] == math.inf:
                r[name]["score"] = "inf"
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    meta = {"seeds": a.seeds, "k": a.k, "modes": a.modes, "n_rows": len(rows),
            "wall_s": round(time.time() - t, 1),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    json.dump({"meta": meta, "rows": rows}, open(a.json, "w"))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
