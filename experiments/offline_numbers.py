"""Offline (no API) numbers for the paper's Appendix A and related sentences.

Stdlib only. Writes runs/references/offline_numbers.json.

  1  detection calibration on held-out seeds (tab:calibration) and
     sensitivity / specificity / d' on the confirmatory seeds (tab:detection)
  2  Bayesian observer variants against the printed bcBayes* cells
  3  sixteen-node family (n_nodes=16, extra_edges=20): eligibility,
     localization curves, routes (tab:n16) and calibration (tab:calibration16)
  4  reference Localization restricted to the queried pairs (app:full)
  5  adaptation lag of the evidence-only reference agent (agentic arm)
  6  transcribed numbers that can be recomputed from runs/

Calibration rule: per reference, K and mode, the threshold t is the smallest
observed held-out no change score with #(no change score > t) <= 10 percent;
an instance is detected when its score > t (menu removal scores inf).
d' is log-linear corrected: H = (hits + 0.5) / (n + 1), likewise F.

Usage: python3 experiments/offline_numbers.py [--out runs/references/offline_numbers.json]
"""
import argparse, functools, glob, json, math, os, random, re, statistics, sys
from multiprocessing import Pool
from statistics import NormalDist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ecpm_baseline as B
import resource_mdp as R
from experiments import reference_sweep as RS
from experiments.baseline_k_sweep import true_pair

REFS = ("rate", "transition", "bayes")
TEX = os.path.join(ROOT, "paper", "tex", "ECPM_main.tex")
REFDIR = os.path.join(ROOT, "runs", "references")
TAG = {"silent_break": "Silentbreak", "degradation": "Degradation", "redirect": "Redirect",
       "irrelevant": "Irrelevant", "hard_removal": "Hardremoval"}


def macros():
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", open(TEX).read()))


def score(r, ref):
    v = r[ref]["score"] if r.get(ref) else None
    return math.inf if v == "inf" else (-math.inf if v is None else v)


def eligible(rows):
    """Seeds where all five change conditions build (stochastic, K = 10)."""
    by = {}
    for r in rows:
        if r["mode"] == "sto" and r["k"] == 10 and r["condition"] != "no_change":
            by.setdefault(r["seed"], set()).add(r["condition"])
    return sorted(s for s, c in by.items() if len(c) == 5)


def threshold(nc):
    for s in sorted(set(nc)):
        if sum(x > s for x in nc) <= 0.1 * len(nc):
            return s


def dprime(h, n_s, fa, n_n):
    z = NormalDist().inv_cdf
    return z((h + 0.5) / (n_s + 1)) - z((fa + 0.5) / (n_n + 1))


def calibration(held, conf, scorer=score):
    """Held-out FA/SB/Deg percent and confirmatory sens/spec/d' per ref, mode, K."""
    eh, ec = set(eligible(held)), set(eligible(conf))
    out = {}
    for ref in REFS:
        for mode in ("sto", "det"):
            for k in (5, 10, 20):
                H = [r for r in held if r["mode"] == mode and r["k"] == k and r["seed"] in eh]
                t = threshold([scorer(r, ref) for r in H if r["condition"] == "no_change"])
                def pct(c):
                    rr = [r for r in H if r["condition"] == c]
                    return round(100 * sum(scorer(r, ref) > t for r in rr) / len(rr), 1) if rr else None
                C = [r for r in conf if r["mode"] == mode and r["k"] == k and r["seed"] in ec]
                ch = [r for r in C if r["condition"] != "no_change"]
                nn = [r for r in C if r["condition"] == "no_change"]
                h = sum(scorer(r, ref) > t for r in ch)
                fa = sum(scorer(r, ref) > t for r in nn)
                out[f"{ref}|{mode}|{k}"] = {
                    "threshold": t, "n_heldout_seeds": len(eh),
                    "FA": pct("no_change"), "SB": pct("silent_break"), "Deg": pct("degradation"),
                    "sens": round(h / len(ch), 2), "spec": round(1 - fa / len(nn), 2),
                    "dprime": round(dprime(h, len(ch), fa, len(nn)), 2),
                    "n_changed": len(ch), "n_no_change": len(nn)}
    return out


# ---------------------------------------------------------------- instances
def _build(a):
    seed, cond, det, k, nn, ee = a
    if det and cond == "degradation":
        return None
    try:
        inst = R.make_pair(seed, cond, deterministic=det, matched=True, n_nodes=nn, extra_edges=ee)
        ev = R.paired_evidence(inst, k=k, max_episodes=300 * max(1, k // 5), horizon=60 * max(1, k // 5))
        rec = R.pair_to_json(inst, ev)
        pv = R.prompt_view(rec, rendering="F2_shuffled", budget_per_pair=k)
    except (ValueError, AssertionError, KeyError, RuntimeError, TypeError):
        return None
    pre, post = RS.periods(pv)
    return {"seed": seed, "condition": cond, "mode": "det" if det else "sto", "k": k,
            "pre": pre, "post": post, "removed": [tuple(p) for p in B.menu_removals(pv)],
            "truth": true_pair(rec), "nodes": list(inst.m0.nodes)}


def outcome_cache(seeds, nn, ee, pool):
    jobs = [(s, c, d, k, nn, ee) for d in (False, True) for k in (5, 10, 20)
            for c in RS.CONDITIONS for s in seeds]
    return [r for r in pool.map(_build, jobs, chunksize=20) if r]


def _row16(a):
    s, c, d, k = a
    if d and c == "degradation":
        return None
    return RS.row(s, c, d, k, B.CALIBRATED_DELTA_THRESHOLD[k])


def _init16(nn, ee):
    R.make_pair = functools.partial(R.make_pair, n_nodes=nn, extra_edges=ee)


# ------------------------------------------------------------ Bayes variants
def _bf(a, b, keys):
    both = {k: a.get(k, 0) + b.get(k, 0) for k in keys}
    return RS.log_dm(a, keys) + RS.log_dm(b, keys) - RS.log_dm(both, keys)


def _drops(d):
    return d.get("drop", 0)


VARIANTS = {
    "union (current)": lambda r, p, a, b: _bf(a, b, sorted(set(a) | set(b))),
    "union+drop": lambda r, p, a, b: _bf(a, b, sorted(set(a) | set(b) | {"drop"})),
    "binary success/drop": lambda r, p, a, b: _bf(
        {"s": sum(a.values()) - _drops(a), "drop": _drops(a)},
        {"s": sum(b.values()) - _drops(b), "drop": _drops(b)}, ["drop", "s"]),
    "all nodes+drop": lambda r, p, a, b: _bf(a, b, sorted(r["nodes"]) + ["drop"]),
    "node's destinations+drop": lambda r, p, a, b: _bf(a, b, sorted(
        {"drop"} | {x for P in (r["pre"], r["post"]) for (n, _), d in P.items() if n == p[0] for x in d})),
    "union, ties to more Period B drops": lambda r, p, a, b: (
        _bf(a, b, sorted(set(a) | set(b))) + 1e-9 * _drops(b)),
}


def bayes_variant_check(rows, held, fam, M):
    ec, eh = set(eligible(rows)), set(eligible(held))
    prefix = "bcBayes" if fam == 8 else "bcSixteenBayes"
    res = {}
    for name, fn in VARIANTS.items():
        def sc(r):
            if r["removed"]:
                return r["removed"][0], math.inf
            s = {p: fn(r, p, r["pre"][p], r["post"][p]) for p in r["pre"] if p in r["post"]}
            if not s:
                return None, -math.inf
            top = RS.ranked(s)[0]
            return top, s[top]
        cells = {}
        for r in rows:
            if r["truth"] is None or (r["mode"] == "sto" and r["seed"] not in ec):
                continue
            c = cells.setdefault((r["mode"], r["condition"], r["k"]), [0, 0])
            c[0] += 1
            c[1] += sc(r)[0] == tuple(r["truth"])
        mism, n = [], 0
        for (mode, cond, k), (tot, hit) in sorted(cells.items()):
            mac = M.get(f"{prefix}{mode.capitalize()}{TAG[cond]}")
            p = dict(re.findall(r"\((\d+),([\d.]+)\)", mac or "")).get(str(k))
            if p is None:
                continue
            n += 1
            if f"{hit / tot:.2f}" != p:
                mism.append(f"{mode} {cond} K{k}: ours {hit / tot:.2f} paper {p}")
        cal = {}
        for k in (5, 10, 20):
            H = [r for r in held if r["mode"] == "sto" and r["k"] == k and r["seed"] in eh]
            S = {id(r): sc(r)[1] for r in H}
            t = threshold([S[id(r)] for r in H if r["condition"] == "no_change"])
            cal[k] = [round(100 * sum(S[id(r)] > t for r in H if r["condition"] == c)
                            / sum(1 for r in H if r["condition"] == c), 1)
                      for c in ("no_change", "silent_break", "degradation")]
        res[name] = {"cells_compared": n, "cells_matching": n - len(mism), "mismatches": mism,
                     "heldout_FA_SB_Deg": cal}
    return res


# ---------------------------------------------------------- queried pairs
def _queried(seed):
    import run_pilot as P
    inst = R.make_pair(seed, "silent_break", deterministic=False, matched=True)
    ev = R.paired_evidence(inst, k=10, max_episodes=600, horizon=120)
    rec = json.loads(json.dumps(R.pair_to_json(inst, ev)))
    pv = R.prompt_view(rec, rendering="F2_shuffled", budget_per_pair=10)
    q = {(d["node"], d["action"]) for d in P.queried_pairs_for_icl(rec, {"seed": seed, "matched": True})}
    truth = true_pair(rec)
    out = {"seed": seed, "target_queried": truth in q}
    ranked = [tuple(x["pair"]) for x in B.rank_pairs(*B.read_periods(pv)[:2]) if tuple(x["pair"]) in q]
    out["rate"] = bool(ranked) and ranked[0] == truth
    pre, post = RS.periods(pv)
    orig = RS.periods
    RS.periods = lambda _pv: ({k: v for k, v in pre.items() if k in q}, {k: v for k, v in post.items() if k in q})
    try:
        refs = RS.references(pv, False)
    finally:
        RS.periods = orig
    for name in ("transition", "bayes", "generator"):
        out[name] = bool(refs.get(name)) and tuple(refs[name]["loc"]) == truth
    return out


# ------------------------------------------------------- reference agent
PI = (2 / 6) / 16
GRID = RS.GRID


def _p_fail_run(n):
    return sum((1 - p) ** n for p in GRID) / len(GRID)


def _post_broken(n):
    return PI / (PI + (1 - PI) * _p_fail_run(n)) if n else 0.0


def _mean_p(s, f):
    w = [p ** s * (1 - p) ** f for p in GRID]
    return sum(p * x for p, x in zip(GRID, w)) / sum(w)


def reference_agent(seed, cfg_eps=4, steps=25):
    """Evidence-only agent: Dijkstra on posterior mean p (weight 1/p), avoids a link whose
    posterior of being broken (from its current run of consecutive failures) exceeds 0.95."""
    try:
        inst = R.make_pair(seed, "silent_break", deterministic=False, matched=True)
    except ValueError:
        return None
    cnt, run = {}, {}

    def p_hat(e):
        s, f = cnt.get(e, (0, 0))
        pb = _post_broken(run.get(e, 0))
        return None if pb > 0.95 else (1 - pb) * _mean_p(s, f)

    def choose(mdp, u):
        dist = {mdp.goal: 0.0}
        changed = True
        while changed:  # Bellman-Ford style relaxation on expected attempts
            changed = False
            for (a, b) in mdp.p:
                ph = p_hat((a, b))
                if ph and b in dist and dist[b] + 1 / ph < dist.get(a, math.inf) - 1e-12:
                    dist[a] = dist[b] + 1 / ph
                    changed = True
        opts = [(1 / p_hat((u, v)) + dist[v], v) for v in mdp.out_edges(u) if p_hat((u, v)) and v in dist]
        return min(opts)[1] if opts else mdp.out_edges(u)[0]

    m1_steps = []
    for phase, mdp in (("m0", inst.m0), ("m1", inst.m1)):
        for ep in range(cfg_eps):
            rng = random.Random(f"{seed}|explore|{phase}|{ep}")
            at, t = inst.start, 0
            while at != mdp.goal and t < steps:
                v = choose(mdp, at)
                nxt, ok = mdp.step(at, v, rng)
                t += 1
                s, f = cnt.get((at, v), (0, 0))
                cnt[(at, v)] = (s + ok, f + (not ok))
                run[(at, v)] = 0 if ok else run.get((at, v), 0) + 1
                if phase == "m1":
                    m1_steps.append((at, v, ok))
                at = nxt
    u, v = inst.change["edge"]
    ff = next((i for i, (a, b, ok) in enumerate(m1_steps) if a == u and b == v and not ok), None)
    if ff is None:
        return {"seed": seed, "lag": None, "first_fail": None}
    last = max(i for i, (a, b, _) in enumerate(m1_steps) if a == u and b == v)
    return {"seed": seed, "lag": max(0, last - ff), "attempts_after_first_fail":
            sum(1 for a, b, _ in m1_steps[ff:] if a == u and b == v)}


# ------------------------------------------------------- halved irrelevant
def _halved(a):
    """Old protocol: stochastic irrelevant halves U* instead of breaking it."""
    seed, k = a
    orig = R.RoutingMDP.set_link_prob

    def halve(self, u, v, new_p, mode="silent"):
        old = self.p[(u, v)]
        return orig(self, u, v, max(0.05, round(old * 0.5, 2)), mode="degrade")
    R.RoutingMDP.set_link_prob = halve
    try:
        inst = R.make_pair(seed, "irrelevant", deterministic=False, matched=True)
        ev = R.paired_evidence(inst, k=k, max_episodes=300 * max(1, k // 5), horizon=60 * max(1, k // 5))
        rec = R.pair_to_json(inst, ev)
        pv = R.prompt_view(rec, rendering="F2_shuffled", budget_per_pair=k)
    except (ValueError, AssertionError):
        return None
    finally:
        R.RoutingMDP.set_link_prob = orig
    truth = true_pair(rec)
    res = B.run_baseline(pv)
    refs = RS.references(pv, False)
    out = {"seed": seed, "rate": tuple(res["localization"] or ()) == truth}
    for n in ("transition", "bayes", "generator"):
        out[n] = bool(refs.get(n)) and tuple(refs[n]["loc"]) == truth
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REFDIR, "offline_numbers.json"))
    a = ap.parse_args(argv)
    M = macros()
    out = {}
    load = lambda pat: [r for f in sorted(glob.glob(os.path.join(REFDIR, pat))) for r in json.load(open(f))["rows"]]
    conf8, held8 = load("part_*.json"), load("heldout.json")

    # 1 calibration and detection, eight nodes
    out["1_calibration_detection_n8"] = calibration(held8, conf8)

    with Pool(4) as pool:
        # 3 sixteen-node rows (cached next to the eight-node parts)
        rows16 = {}
        for tag, lo, hi in (("conf", 31, 130), ("held", 1000, 1099)):
            path = os.path.join(REFDIR, f"n16_{tag}.json")
            if not os.path.exists(path):
                with Pool(4, initializer=_init16, initargs=(16, 20)) as p16:
                    jobs = [(s, c, d, k) for d in (False, True) for k in (5, 10, 20)
                            for c in RS.CONDITIONS for s in range(lo, hi + 1)]
                    rr = [r for r in p16.map(_row16, jobs, chunksize=10) if r]
                for r in rr:
                    for n in ("rate", "transition", "bayes", "generator"):
                        if r[n] and r[n]["score"] == math.inf:
                            r[n]["score"] = "inf"
                json.dump({"meta": {"n_nodes": 16, "extra_edges": 20, "seeds": f"{lo}-{hi}"}, "rows": rr}, open(path, "w"))
            rows16[tag] = json.load(open(path))["rows"]
        c16, h16 = rows16["conf"], rows16["held"]
        e16 = set(eligible(c16))
        n16 = {"setting": {"n_nodes": 16, "extra_edges": 20, "hook": "resource_mdp.make_pair(n_nodes=, extra_edges=)"},
               "n_eligible_31_130": len(e16), "n_eligible_heldout": len(eligible(h16)),
               "curves": {}, "checks": []}
        for mode in ("sto", "det"):
            for cond, tag in TAG.items():
                for ref, Rn in (("rate", "Rate"), ("transition", "Trans"), ("bayes", "Bayes"), ("generator", "Gen")):
                    for k in (5, 10, 20):
                        rr = [r for r in c16 if r["mode"] == mode and r["condition"] == cond and r["k"] == k
                              and (mode == "det" or r["seed"] in e16)]
                        if not rr:
                            continue
                        v = f"{sum(r[ref]['loc'] == r['truth'] for r in rr) / len(rr):.2f}"
                        n16["curves"][f"{ref}|{mode}|{cond}|{k}"] = v
                        p = dict(re.findall(r"\((\d+),([\d.]+)\)", M.get(f"bcSixteen{Rn}{mode.capitalize()}{tag}", ""))).get(str(k))
                        if p is not None:
                            n16["checks"].append([f"bcSixteen{Rn}{mode.capitalize()}{tag}@K{k}", v, p, v == p])
        for cond in ("no_change", "silent_break"):
            rr = [r for r in c16 if r["mode"] == "sto" and r["condition"] == cond and r["k"] == 10 and r["seed"] in e16]
            n16[f"route_{cond}"] = f"{sum(r['route_optimal'] for r in rr) / len(rr):.2f}"
        n16["calibration"] = calibration(h16, c16)
        out["3_sixteen_nodes"] = n16

        # 2 Bayes variants, both families
        c8o, h8o = outcome_cache(range(31, 131), 8, 6, pool), outcome_cache(range(1000, 1100), 8, 6, pool)
        out["2_bayes_variants_n8"] = bayes_variant_check(c8o, h8o, 8, M)
        c16o, h16o = outcome_cache(range(31, 131), 16, 20, pool), outcome_cache(range(1000, 1100), 16, 20, pool)
        out["2_bayes_variants_n16"] = bayes_variant_check(c16o, h16o, 16, M)

        # 4 queried pairs, stochastic silent break, K = 10, eight-node run set
        e8 = eligible(conf8)
        q = pool.map(_queried, e8)
        out["4_queried_pairs_sb_sto_k10"] = {
            "selection": "run_pilot.queried_pairs_for_icl (target always among the five)",
            "n": len(q), "target_queried": sum(x["target_queried"] for x in q),
            **{n: f"{sum(x[n] for x in q) / len(q):.2f}" for n in ("rate", "transition", "bayes", "generator")}}

        # 5 reference agent adaptation lag, agentic arm seeds
        import agentic_conditions as AC
        ag = [x for x in pool.map(reference_agent, AC.SEEDS_51) if x]
        lags = [x["lag"] for x in ag if x["lag"] is not None]
        out["5_reference_agent_lag"] = {
            "seeds": len(AC.SEEDS_51), "built": len(ag), "n_with_failure": len(lags),
            "median": statistics.median(lags) if lags else None,
            "mean": round(statistics.mean(lags), 2) if lags else None,
            "min": min(lags) if lags else None, "max": max(lags) if lags else None,
            "posterior_after_n_failures": {n: round(_post_broken(n), 3) for n in (4, 5, 6)}}

        # 6 transcribed numbers
        sweep = json.load(open(os.path.join(ROOT, "runs", "baseline_k_sweep_seeds1-30.json")))
        fa = next(r for r in sweep["rows"] if r["mode"] == "sto" and r["condition"] == "no_change" and r["k"] == 10)
        hv = {}
        for lo, hi, name in ((1, 30, "seeds_1_30"), (31, 130, "seeds_31_130")):
            rr = [x for x in pool.map(_halved, [(s, 10) for s in range(lo, hi + 1)]) if x]
            if name == "seeds_31_130":
                rr = [x for x in rr if x["seed"] in set(e8)]
            hv[name] = {"n": len(rr), **{n: round(100 * sum(x[n] for x in rr) / len(rr), 1)
                                          for n in ("rate", "transition", "bayes", "generator")}}
        out["6_transcribed"] = {
            "pctOldFalseAlarm": {"value": round(100 * fa["false_alarm_rate"]), "n": fa["n"],
                                 "source": "runs/baseline_k_sweep_seeds1-30.json, sto no_change K=10, threshold 0.50"},
            "pctHalved_localization_pct_K10": hv,
            "pilotRepeatPct": "not recomputable: no GPT-5.6 Sol seed 7 agentic artifact in runs/",
            "pilotLag": "not recomputable: no GPT-5.6 Sol seed 7 agentic artifact in runs/"}

    def clean(o):
        if isinstance(o, float) and math.isinf(o):
            return "inf"
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    json.dump(clean(out), open(a.out, "w"), indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
