#!/usr/bin/env python3
"""Generate graph-availability previews and reference checks without a provider."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import icl_graph as graph
import run_pilot as pilot


def eligibility_report():
    """Apply the existing deterministic silent-break rule, without new filters."""
    checks = [pilot.deterministic_gate(seed) for seed in range(1, 1001)]
    eligible = [r["seed"] for r in checks if r["eligible"]]
    if eligible[:3] != list(graph.GRAPH_SEEDS):
        raise ValueError("deterministic eligibility selection drifted")
    return {"rule": "run_pilot.deterministic_gate, unchanged", "range": [1, 1000],
            "n_examined": len(checks), "eligible_seeds": eligible,
            "first_three": eligible[:3], "checks_through_seed25": checks[:25]}


def generate(out, input_price=None, output_price=None, seed=8):
    out = Path(out)
    if out.exists():
        raise ValueError("preview directory exists; use an unused path")
    sc = dict(pilot.SCENARIO_DEFAULTS)
    sc.update(pilot.SCENARIOS[f"icl_det_gate_seed{seed}"])
    sc["name"] = f"icl_det_gate_seed{seed}"
    record, view, variants, prompts = graph.prepare(sc)
    queried = pilot.queried_pairs_for_icl(record, sc)
    target = pilot.protocol_target_pair(record, sc)
    report = {"protocol": graph.PROTOCOL, "implementation_base": pilot.git_head(),
              "model_calls": 0, "graph_seed": seed, "evidence_seed": 0,
              "start": view["start"], "goal": view["goal"],
              "target": target, "queried_pairs": queried,
              "first_eligible_seed": pilot.first_deterministic_gate_seed(),
              "prompts": {}, "references": {}, "context_planning": {},
              "request_profiles": {}, "sol_cost_estimate": {}}
    assumption = {"context": {"method": "utf8_upper_bound", "tokens": 32768,
                  "overhead_tokens_per_message": 256, "replay_expansion_bound": 1,
                  "source": "PLANNING ASSUMPTION ONLY: token<=UTF8 bytes, <=256 template tokens/message, A replay<=8192 tokens; must verify for each endpoint"}}
    for condition in graph.CONDITIONS:
        reference = {}
        earlier = pre_score = None
        for i, period in enumerate(("A", "B")):
            v = variants[condition][i]
            text = prompts[condition][i]
            raw, earlier = graph.reference_answer(v, earlier)
            parsed = (pilot.parse_icl_turn_a if i == 0 else pilot.parse_icl_turn_b)(raw)
            scored = pilot.score_icl_turn(record, parsed, queried, "pre" if i == 0 else "post",
                                         target, pilot.visible_transition_stats(v["rows"], v["menu"]), pre_score)
            if not scored["correct"]:
                raise ValueError("sequential reference did not pass unchanged scorer")
            pre_score = scored["beliefs"]
            reference[period] = {"raw_answer": raw, "parsed": parsed, "scored": scored}
            report["prompts"][condition + "_" + period] = {
                "sha256": pilot.sha256_text(text), "utf8_bytes": len(text.encode()),
                "observation_count": len(v["rows"]),
                "rows_sha256": pilot.sha256_text("\n".join(v["rows"]) + "\n"),
                "non_graph_sha256": pilot.sha256_text(text.replace(graph.graph_block(v), "", 1)
                                                      if v["graph"] is not None else text)}
        report["references"][condition] = reference
        messages = [{"role": "user", "content": prompts[condition][0]},
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": prompts[condition][1]}]
        planning = graph.context_bound(messages, assumption, reserve_answer=8192)
        planning["verified_for_deployment"] = False
        report["context_planning"][condition] = planning
        if input_price is not None and output_price is not None:
            a_input = len(prompts[condition][0].encode()) + 256
            input_total = 3 * (a_input + planning["input_tokens_upper_bound"])
            output_total = 6 * 8192
            report["sol_cost_estimate"][condition] = {
                "runs": 3, "answers": 6, "input_tokens_planning_bound": input_total,
                "output_tokens_allowance": output_total,
                "input_usd_per_million": input_price, "output_usd_per_million": output_price,
                "usd": (input_total * input_price + output_total * output_price) / 1e6,
                "assumptions": "uncached input; every answer consumes all 8192 tokens; same unverified context bounds; preflight excluded"}
    for profile, model in graph.MODELS.items():
        report["request_profiles"][profile] = {
            "recommended_model_not_a_guessed_deployment": model,
            "endpoint_verified": False, "effective_controls_verified": False,
            "intended_bodies": {mode: {"model": model,
                "messages": [{"role": "user", "content": prompts["graph_ab"][0]}],
                **graph.intended_settings(profile, mode, 0, profile != "sol")}
                for mode in ("off", "on")},
            "seed_note": "Sol omits seeds; Gemma example assumes supported, otherwise omit seed and record unsupported",
        }
    out.mkdir(parents=True, exist_ok=False)
    for condition, texts in prompts.items():
        for period, text in zip(("A", "B"), texts):
            (out / f"{condition}_{period}.txt").write_text(text, encoding="utf-8")
    (out / "offline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, choices=graph.GRAPH_SEEDS, default=8)
    ap.add_argument("--eligibility-report", help="save the unchanged seed-search audit outside Git")
    ap.add_argument("--input-price", type=float)
    ap.add_argument("--output-price", type=float)
    args = ap.parse_args()
    if (args.input_price is None) != (args.output_price is None):
        ap.error("supply both prices or neither")
    if args.eligibility_report:
        with Path(args.eligibility_report).open("x") as f:
            json.dump(eligibility_report(), f, indent=2)
            f.write("\n")
    report = generate(args.out, args.input_price, args.output_price, args.seed)
    print(json.dumps({k: report[k] for k in ("model_calls", "graph_seed", "context_planning", "sol_cost_estimate")}, indent=2))
