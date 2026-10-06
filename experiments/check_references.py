"""Aggregate reference_sweep rows and compare with the numbers printed in ECPM_main.tex.

Exit 0: every comparable cell matches. Exit 1: a mismatch (each is named).
Exit 2: could not run (no rows, or no macro found to compare against).

Usage: python3 experiments/check_references.py ROWS_DIR TEX [--out summary.json]
"""
import glob, json, math, os, re, sys

COND = {"silent_break": "SB", "degradation": "Deg", "redirect": "Red"}
KW = {5: "Five", 10: "Ten", 20: "Twenty"}
CURVE = {"silent_break": "Silentbreak", "degradation": "Degradation", "redirect": "Redirect",
         "irrelevant": "Irrelevant", "hard_removal": "Hardremoval"}


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "part_*.json"))):
        rows += json.load(open(f))["rows"]
    return rows


def aggregate(rows):
    sto_change = [r for r in rows if r["mode"] == "sto" and r["k"] == 10 and r["condition"] != "no_change"]
    by = {}
    for r in sto_change:
        by.setdefault(r["seed"], set()).add(r["condition"])
    need = {"irrelevant", "silent_break", "hard_removal", "redirect", "degradation"}
    eligible = sorted(s for s, c in by.items() if need <= c)
    out = {"n_seeds_sto": len(eligible), "cells": {}}
    for r in rows:
        if r["mode"] == "sto" and r["seed"] not in eligible:
            continue
        key = f'{r["mode"]}|{r["condition"]}|{r["k"]}'
        c = out["cells"].setdefault(key, {"n": 0, "route_optimal": 0, "cpu_s": 0.0,
                                          **{f"loc_{n}": 0 for n in ("rate", "transition", "bayes", "generator")}})
        c["n"] += 1
        c["route_optimal"] += r["route_optimal"]
        c["cpu_s"] += r["cpu_s"]["rate"] + r["cpu_s"]["others"]
        if r["truth"]:
            for n in ("rate", "transition", "bayes", "generator"):
                c[f"loc_{n}"] += (r[n]["loc"] == r["truth"])
    return out


def macros(tex):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", tex))


def compare(agg, tex):
    m = macros(tex)
    checks = []
    def add(name, ours, theirs):
        checks.append((name, ours, theirs, ours == theirs))
    for cond, tag in CURVE.items():
        for mode, M in (("sto", "Sto"), ("det", "Det")):
            for ref, R in (("rate", "Rate"), ("transition", "Trans"), ("bayes", "Bayes")):
                mac = m.get(f"bc{R}{M}{tag}")
                if not mac:
                    continue
                for k, p in re.findall(r"\((\d+),([\d.]+)\)", mac):
                    c = agg["cells"].get(f"{mode}|{cond}|{k}")
                    if c:
                        add(f"bc{R}{M}{tag}@K{k}", f'{c[f"loc_{ref}"] / c["n"]:.2f}', p)
        mac = m.get(f"bcGenSto{tag}")
        if mac:
            for k, p in re.findall(r"\((\d+),([\d.]+)\)", mac):
                c = agg["cells"].get(f"sto|{cond}|{k}")
                if c:
                    add(f"bcGenSto{tag}@K{k}", f'{c["loc_generator"] / c["n"]:.2f}', p)
    for cond, tag in (("silent_break", "SB"), ("degradation", "Deg")):
        c = agg["cells"].get(f"sto|{cond}|10")
        if c and f"routeRef{tag}" in m:
            add(f"routeRef{tag}", f'{c["route_optimal"] / c["n"]:.2f}', m[f"routeRef{tag}"].strip())
    if "nSeedsEight" in m:
        add("nSeedsEight", str(agg["n_seeds_sto"]), m["nSeedsEight"].strip())
    return checks


def main(a):
    if len(a) < 2:
        print(__doc__); return 2
    rows = load(a[0])
    tex = open(a[1]).read() if os.path.exists(a[1]) else ""
    if not rows or not tex:
        print("COULD NOT RUN: 0 rows or no tex"); return 2
    agg = aggregate(rows)
    checks = compare(agg, tex)
    if not checks:
        print("COULD NOT RUN: no comparable macro found"); return 2
    bad = [c for c in checks if not c[3]]
    for name, ours, theirs, ok in checks:
        print(f'{"match   " if ok else "MISMATCH"} {name:32s} ours={ours} paper={theirs}')
    print(f"{len(checks) - len(bad)} of {len(checks)} cells match")
    if "--out" in a:
        json.dump({"aggregate": agg, "checks": checks}, open(a[a.index("--out") + 1], "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
