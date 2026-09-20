#!/usr/bin/env python3
"""Acceptance test for the graph ICL levels.

The six files in the collaborator's `prompts/` directory are the contract:
they are the exact texts his reviewed implementation produced. This asserts
that build_icl_graph_prompt reproduces all six byte for byte from the frozen
scenario, so any later edit to the wording is a loud failure rather than a
silent divergence between his review and what we actually send.

Run from the repository root:

    python3 test_icl_graph_prompts.py --prompts /path/to/prompts
"""

import argparse
import difflib
import os
import sys

import run_pilot

SCENARIO = "icl_det_gate_seed8"
DETERMINISTIC = True
FILES = {("graph_ab", "pre"): "graph_ab_A.txt",
         ("graph_ab", "post"): "graph_ab_B.txt",
         ("graph_a", "pre"): "graph_a_A.txt",
         ("graph_a", "post"): "graph_a_B.txt",
         ("logs_only", "pre"): "logs_only_A.txt",
         ("logs_only", "post"): "logs_only_B.txt"}


def build():
    sc = dict(run_pilot.SCENARIO_DEFAULTS)
    sc.update(run_pilot.SCENARIOS[SCENARIO])
    sc["name"] = SCENARIO
    record = run_pilot.build_record(sc, DETERMINISTIC)
    view = run_pilot.prompt_view(record, rendering="F2_shuffled",
                                 periods=("pre", "post"),
                                 budget_per_pair=sc["budget"], budget_seed=0)
    queried = run_pilot.queried_pairs_for_icl(record, sc)
    return record, view, queried


def test_scenario_matches_the_reviewed_design(view, queried):
    """The design fixes the queried pairs, the target and the log size."""
    assert [(q["node"], q["action"]) for q in queried] == [
        ("B", "a2"), ("F", "a1"), ("F", "a2"), ("G", "a1"), ("H", "a2")]
    for period in ("pre", "post"):
        assert len(run_pilot.raw_visible_rows(view, period)) == 160
    print("PASS five queried pairs and 160 observations per period")


def test_only_the_graph_block_differs(record, view, queried):
    """Levels must differ in the graph block alone, never in wording."""
    def lines(level, period):
        return run_pilot.build_icl_graph_prompt(
            record, view, level, period, queried).splitlines()

    for period in ("pre", "post"):
        bare = lines("logs_only", period)
        for level in ("graph_ab", "graph_a"):
            richer = lines(level, period)
            removed = [row for row in difflib.unified_diff(
                richer, bare, lineterm="", n=0) if row.startswith("-")
                and not row.startswith("---")]
            added = [row for row in difflib.unified_diff(
                richer, bare, lineterm="", n=0) if row.startswith("+")
                and not row.startswith("+++")]
            assert not added, f"{level} {period} drops text present in logs_only"
            body = [row[1:] for row in removed if row[1:].strip()]
            if body:
                assert body[0].startswith("Complete graph, Period"), body[0]
                assert body[1].startswith("node | action |"), body[1]
    assert lines("graph_ab", "pre") == lines("graph_a", "pre")
    assert lines("graph_a", "post") == lines("logs_only", "post")
    print("PASS levels differ only by the complete-graph block")


def test_reproduces_supplied_prompts(record, view, queried, directory):
    failures = 0
    for (level, period), name in sorted(FILES.items()):
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            expected = handle.read()
        actual = run_pilot.build_icl_graph_prompt(
            record, view, level, period, queried)
        if actual == expected:
            continue
        failures += 1
        print(f"FAIL {name}")
        diff = difflib.unified_diff(expected.splitlines(),
                                    actual.splitlines(),
                                    "supplied", "generated", lineterm="", n=1)
        print("\n".join(list(diff)[:40]))
    if failures:
        raise AssertionError(f"{failures} of {len(FILES)} prompts differ")
    print(f"PASS all {len(FILES)} supplied prompts reproduce byte for byte")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts",
                    help="directory holding the six supplied .txt prompts")
    args = ap.parse_args()
    if not os.path.isdir(args.prompts):
        sys.exit(f"no such directory: {args.prompts}")
    record, view, queried = build()
    test_scenario_matches_the_reviewed_design(view, queried)
    test_only_the_graph_block_differs(record, view, queried)
    test_reproduces_supplied_prompts(record, view, queried, args.prompts)


if __name__ == "__main__":
    main()
