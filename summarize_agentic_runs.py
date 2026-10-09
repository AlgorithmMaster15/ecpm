"""Summarise agentic runs for the Results doc. Stdlib only.

    python summarize_agentic_runs.py RUN_DIR

Writes RUN_DIR/agentic_runs.csv (one row per run, filterable: seed, mode, scenario, arm,
history, repeat, model, every metric) and RUN_DIR/agentic_summary.md (mean and spread
per arm, ready to paste into Google Docs as Markdown or into Slack as a code block).
"""
import csv, glob, json, os, statistics as st, sys

def load(run_dir):
    rows = []
    for meta_p in sorted(glob.glob(os.path.join(run_dir, "**", "condition.json"), recursive=True)):
        d = os.path.dirname(meta_p)
        arts = glob.glob(os.path.join(d, "pilot_*.json"))
        if not arts:
            continue
        a, meta = json.load(open(arts[0], encoding="utf-8")), json.load(open(meta_p, encoding="utf-8"))
        ins, m, p = a["instance"], a["explore"]["metrics"], a.get("probes", {})
        ex = meta.get("exposure", {})
        tok = a.get("token_usage_total") or {}
        sc = lambda k: (p.get(k, {}).get("scored") or {})
        rows.append({
            "run": os.path.basename(d), "seed": ins["graph_seed"], "mode": "det" if ins.get("deterministic") else "sto",
            "scenario": ins["condition"], "arm": meta.get("condition"), "history": meta.get("history", "none"),
            "matched_prep": meta.get("matched_prep", False), "repeat": meta.get("repeat", 0),
            "model": (a.get("model") or {}).get("model", "") if isinstance(a.get("model"), dict) else a.get("model", ""),
            "goal_success_m0": m.get("goal_success_rate_m0"), "goal_success_m1": m.get("goal_success_rate_m1"),
            "steps_m0": (m.get("steps_to_goal_m0") or {}).get("mean"), "steps_m1": (m.get("steps_to_goal_m1") or {}).get("mean"),
            "optimal_action_m0": m.get("optimal_action_rate_m0"), "optimal_action_m1": m.get("optimal_action_rate_m1"),
            "regret_m0": m.get("route_regret_m0"), "regret_m1": m.get("route_regret_m1"),
            "exposed": ex.get("exposed"), "must_update": ex.get("must_update"), "m1_attempts": ex.get("m1_attempts"),
            "detection": sc("detection").get("correct"), "localization": sc("localization").get("correct"),
            "preservation": sc("preservation").get("accuracy"),
            "route_probe_optimal": (sc("adaptation").get("regret") == 0) if "regret" in sc("adaptation") else None,
            "reasoning": (a.get("reasoning_control") or {}).get("mode", "unrecorded"),
            "reasoning_tokens": (a.get("reasoning_control") or {}).get("reasoning_tokens_total"),
            "reasoning_violation": (a.get("reasoning_control") or {}).get("violation"),
            "billed_cost": (sum(u["cost"] for u in a.get("provider_usage_calls") or [])
                            if a.get("provider_usage_calls") and all(isinstance((u or {}).get("cost"), (int, float))
                                                                     for u in a["provider_usage_calls"]) else None),
            "calls": tok.get("n_calls"), "input_tokens": tok.get("prompt_tokens"), "output_tokens": tok.get("completion_tokens"),
        })
    return rows

PRICE_IN, PRICE_OUT = 2.50, 10.00   # USD per million tokens (gpt-4o list price; check yours)

def num(v):
    if isinstance(v, bool): return float(v)
    return float(v) if isinstance(v, (int, float)) else None

def agg(vals):
    xs = [x for x in map(num, vals) if x is not None]
    if not xs: return "n/a"
    f = (lambda x: f"{x:,.0f}") if min(xs) >= 100 else (lambda x: f"{x:.2f}")
    return f(st.mean(xs)) if len(xs) == 1 else f"{f(st.mean(xs))} ± {f(st.stdev(xs))}"

def tables(rows):
    lines = []
    arms = [a for a in ("task_only", "model_first", "graph_given") if any(r["arm"] == a for r in rows)]
    by = {a: [r for r in rows if r["arm"] == a] for a in arms}
    metrics = [("Runs (seeds)", None), ("Goal success M1", "goal_success_m1"), ("Mean steps M0", "steps_m0"),
               ("Mean steps M1", "steps_m1"), ("Optimal action rate M0", "optimal_action_m0"),
               ("Optimal action rate M1", "optimal_action_m1"), ("Route regret M0", "regret_m0"),
               ("Exposed (share of runs)", "exposed"), ("Detection correct", "detection"),
               ("Localization correct", "localization"), ("Preservation accuracy", "preservation"),
               ("Route probe optimal", "route_probe_optimal"), ("Input tokens", "input_tokens"),
               ("Output tokens", "output_tokens"), ("Cost per run (USD, billed where reported)", "cost_usd")]
    lines += ["| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for label, key in metrics:
        if key is None:
            cells = [f"{len(by[a])} ({', '.join(str(r['seed']) for r in by[a])})" for a in arms]
        else:
            cells = [agg(r[key] for r in by[a]) for a in arms]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    exp = [r for r in rows if r["exposed"] is True]
    lines += ["", "Values are mean ± standard deviation across runs. Exposed runs only "
              f"({len(exp)} of {len(rows)}):", ""]
    lines += ["| Metric | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for label, key in (("Runs", None), ("Detection correct", "detection"), ("Localization correct", "localization"),
                       ("Preservation accuracy", "preservation")):
        sub = {a: [r for r in by[a] if r["exposed"] is True] for a in arms}
        cells = [str(len(sub[a])) if key is None else agg(r[key] for r in sub[a]) for a in arms]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines

def main(run_dir):
    rows = load(run_dir)
    if not rows:
        sys.exit("no runs found (expected */condition.json and */pilot_*.json)")
    for r in rows:
        if r.get("billed_cost") is None:   # no provider-reported cost: estimate from list prices
            r["cost_usd"], r["cost_source"] = round((r["input_tokens"] or 0) / 1e6 * PRICE_IN + (r["output_tokens"] or 0) / 1e6 * PRICE_OUT, 4), "estimate"
        else:
            r["cost_usd"], r["cost_source"] = round(r["billed_cost"], 4), "billed"
    with open(os.path.join(run_dir, "agentic_runs.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = []
    # one table set per combination of the settings that vary across the folder
    keys = [k for k in ("scenario", "mode", "model", "reasoning", "history") if len({r[k] for r in rows}) > 1]
    groups = sorted({tuple(r[k] for k in keys) for r in rows})
    for g in groups:
        sub = [r for r in rows if tuple(r[k] for k in keys) == g]
        if keys:
            lines += [f"### {' · '.join(f'{k} {v}' for k, v in zip(keys, g))} ({len(sub)} runs)", ""]
        lines += tables(sub) + [""]
    out = os.path.join(run_dir, "agentic_summary.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:   # some Windows consoles cannot print "±"; the files are still UTF-8
        print("\n".join(lines).encode("ascii", "replace").decode())
    print(f"\nwrote {out} and agentic_runs.csv ({len(rows)} runs)")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
