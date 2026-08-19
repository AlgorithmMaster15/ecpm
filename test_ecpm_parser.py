"""Acceptance tests for the frozen Section 7 parser/scorer (ECPM v2.1).

Run:  python3 test_ecpm_parser.py     (stdlib only)

Covers:
  - format-level cases from parser_fixtures.json (json extraction,
    invalid objects, too-long routes, code fences, extra fields)
  - instance-level cases built on the seed-7 records: optimal route,
    zero-regret scoring, stale route through a silent break, stale route
    through a hard removal, discontinuity, unknown label, incomplete
    route, detection/localization/preservation ground truth
"""

import json

from ecpm_parser import (PARSERS, run_probe)
from resource_mdp import make_pair, pair_to_json


def record_for(condition, deterministic=True):
    inst = make_pair(7, condition, deterministic=deterministic)
    return json.loads(json.dumps(pair_to_json(inst)))


def steps_from(record, which="post"):
    o = record["oracle"][which]
    return [{"node": n, "action": a}
            for n, a in zip(o["optimal_route"], o["optimal_actions"])]


def test_format_fixtures():
    with open("parser_fixtures.json") as fh:
        cases = json.load(fh)["cases"]
    n = 0
    for case in cases:
        if "raw_route_repeat" in case:
            raw = json.dumps({"route": [case["raw_route_repeat"]]
                              * case["raw_route_repeat_n"]})
        else:
            raw = case["raw"]
        parsed = PARSERS[case["probe"]](raw)
        assert parsed["status"] == case["expect_status"], \
            f"{case['probe']}: {parsed['status']} != {case['expect_status']}"
        for key, val in case.get("expect", {}).items():
            assert parsed[key] == val
        n += 1
    print(f"PASS format fixtures ({n} cases)")


def test_adaptation_scoring():
    rec = record_for("silent_break")
    # optimal route -> valid_finite, regret 0, is_optimal
    ok = run_probe(rec, "adaptation",
                   json.dumps({"route": steps_from(rec, "post")}))
    assert ok["scored"]["status"] == "valid_finite"
    assert ok["scored"]["is_optimal"] and abs(ok["scored"]["regret"]) < 1e-9
    # stale (pre-optimal) route -> silent_broken_edge
    stale = run_probe(rec, "adaptation",
                      json.dumps({"route": steps_from(rec, "pre")}))
    assert stale["scored"]["status"] == "silent_broken_edge"
    assert stale["scored"]["expected_cost"] is None
    # same stale route on the matched hard removal -> illegal_action
    hard = record_for("hard_removal")
    ill = run_probe(hard, "adaptation",
                    json.dumps({"route": steps_from(hard, "pre")}))
    assert ill["scored"]["status"] == "illegal_action"
    # discontinuity
    broken = steps_from(rec, "post")
    assert len(broken) >= 2, "seed-7 route long enough for the test"
    broken[1] = {"node": rec["goal"], "action": broken[1]["action"]}
    disc = run_probe(rec, "adaptation", json.dumps({"route": broken}))
    assert disc["scored"]["status"] == "discontinuous_route"
    # unknown label at the start node
    unk = run_probe(rec, "adaptation", json.dumps(
        {"route": [{"node": rec["start"], "action": "a99"}]}))
    assert unk["scored"]["status"] == "unknown_reference"
    # incomplete route (drop the last hop)
    inc = run_probe(rec, "adaptation",
                    json.dumps({"route": steps_from(rec, "post")[:-1]}))
    assert inc["scored"]["status"] == "incomplete_route"
    print("PASS adaptation scoring: optimal / silent / illegal / "
          "discontinuous / unknown / incomplete all distinct")


def test_other_probes():
    rec = record_for("silent_break")
    d = run_probe(rec, "detection", '{"changed": true}')
    assert d["scored"]["correct"] is True
    d2 = run_probe(rec, "detection", '{"changed": false}')
    assert d2["scored"]["correct"] is False
    nc = record_for("no_change", deterministic=False)
    d3 = run_probe(nc, "detection", '{"changed": false}')
    assert d3["scored"]["correct"] is True
    # localization ground truth comes from the exact diff
    ch = rec["change"]
    loc = run_probe(rec, "localization", json.dumps(
        {"node": ch["edge"]["from"], "action": ch["action"]}))
    assert loc["scored"]["correct"] is True
    loc2 = run_probe(rec, "localization", json.dumps(
        {"node": ch["edge"]["from"], "action": "a99"}))
    assert loc2["scored"]["correct"] is False
    # preservation: unchanged pair vs the target pair
    target = (ch["edge"]["from"], ch["action"])
    other = next((e["from"], e["action"])
                 for e in rec["world_pre"]["edges"]
                 if (e["from"], e["action"]) != target)
    raw = json.dumps({"pairs": [
        {"node": other[0], "action": other[1], "changed": False},
        {"node": target[0], "action": target[1], "changed": True}]})
    pres = run_probe(rec, "preservation", raw)
    assert pres["scored"]["accuracy"] == 1.0
    queried = [{"node": other[0], "action": other[1]}]
    pres2 = run_probe(rec, "preservation", raw, queried_pairs=queried)
    assert pres2["scored"]["n_scored"] == 1
    print("PASS detection / localization / preservation ground truth")


if __name__ == "__main__":
    test_format_fixtures()
    test_adaptation_scoring()
    test_other_probes()
    print("\nALL PARSER TESTS PASSED")
