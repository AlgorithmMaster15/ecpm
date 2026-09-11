"""Acceptance tests for coherence.py (compression + distinction-recall
world-model coherence probes for the active-exploration pilot).

Run:  python3 test_coherence.py     (stdlib only, no network)

Covers, per docs/coherence_probes_implementation_plan.txt Part 4:
  - build_dfa_view / dfa_step on the seed-7 deterministic pair: goal has
    no menu and rejects every symbol; a silently broken link self-loops;
    a hard-removed label rejects.
  - minimize on a hand-built graph with two intentionally equivalent
    nodes.
  - myhill_nerode_boundary: minimality, directedness (one direction can
    be empty), q1==q2 always empty (on real seed-7 data), and respecting
    the length cap on a cyclic instance.
  - visited_nodes / visited_histories and pair selection (compression +
    distinction) on real dry-run episodes, including the goal-node,
    single-history (-> alternate_history), unvisited-node, and
    Myhill-Nerode-equivalent skips.
  - sample_distinction_recall_targets: the round-robin split (including
    the asymmetric-at-odd-total case), one direction with no boundary
    words at all, and the no-filler-when-undersupplied case.
  - accepted_set / score_compression / score_distinction_recall,
    including the two negative-control fakes (always-start-node stays
    consistent despite being wrong; last-action-label genuinely falsifies
    compression), a known-value per-direction recall case, the
    zero-evaluable-words -> None case, and the missing-answer exclusion.
  - diagnostics_for_pair on a small hand-built answer set with a known
    expected result.
  - parse_validity status coverage, including realistic messy model
    output (reasoning preamble, code fence).
  - End-to-end dry run on a real seed-7 instance: the oracle scores 1.0
    (or leaves recall undefined where nothing was evaluable) throughout.
  - Property checks across ~60 real generated instances (not just the
    hand-built fixtures above): myhill_nerode_boundary matches an
    independent brute-force reference exactly, and every multi-member
    minimize() class has an identical accepted language across its
    members.
"""

import random
from itertools import product

from coherence import (accepted_set, alternate_history, build_dfa_view,
                       diagnostics_for_pair, dfa_step, dry_run_legality_reply,
                       expand_with_prefixes, is_legal_from, language_upto,
                       make_dry_run_answer_fn, minimize,
                       myhill_nerode_boundary, parse_validity,
                       run_sequence, sample_compression_pairs,
                       sample_compression_targets, sample_distinction_pairs,
                       sample_distinction_recall_targets, score_compression,
                       score_distinction_recall, shortest_histories,
                       true_labels, visited_histories, visited_nodes)
from explore_agent import (EpisodeOutcome, ExploreConfig, LiveStep,
                           dry_run_policy, run_explore_instance)
from resource_mdp import RoutingMDP, make_pair


# --------------------------------------------------------------------------
# DFA construction and one-step transitions
# --------------------------------------------------------------------------


def test_dfa_step_goal_silent_break_hard_removal():
    """On the real seed-7 deterministic pair: the goal has no menu entry
    and rejects every symbol; a silently broken link is a self-loop and
    stays accepted; a hard-removed label rejects."""
    inst = make_pair(7, "silent_break", deterministic=True, matched=True)
    dfa0 = build_dfa_view(inst.m0, inst.labels)
    assert inst.m0.goal not in dfa0.menu
    assert dfa_step(dfa0, inst.m0.goal, "a1") is None

    dfa1 = build_dfa_view(inst.m1, inst.labels)
    u, v = inst.m1.changes[0]["edge"]
    lab = inst.labels[(u, v)]
    assert dfa_step(dfa1, u, lab) == u, "silent break must self-loop"

    inst_hr = make_pair(7, "hard_removal", deterministic=True, matched=True)
    dfa_hr1 = build_dfa_view(inst_hr.m1, inst_hr.labels)
    ru, rv = inst_hr.m1.changes[0]["edge"]
    rlab = inst_hr.labels[(ru, rv)]
    assert dfa_step(dfa_hr1, ru, rlab) is None, "hard-removed label must reject"
    print("PASS dfa_step (goal reject / silent-break self-loop / hard-removal reject)")


def test_run_sequence_reject_is_absorbing():
    """Once a sequence hits reject, every further step also rejects, and
    the empty sequence is vacuously legal from any node."""
    mdp = RoutingMDP(["A", "B", "G"], "G", {("A", "B"): 1.0, ("B", "G"): 1.0})
    labels = {("A", "B"): "a1", ("B", "G"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    assert run_sequence(dfa, "A", ()) == "A"
    assert run_sequence(dfa, "A", ("a1", "a1")) == "G"
    assert run_sequence(dfa, "A", ("a2",)) is None       # a2 illegal at A
    assert run_sequence(dfa, "A", ("a2", "a1")) is None   # stays rejected
    assert is_legal_from(dfa, "A", ())
    assert not is_legal_from(dfa, "A", ("a2",))
    print("PASS run_sequence / is_legal_from (reject is absorbing)")


# --------------------------------------------------------------------------
# minimize (exact Myhill-Nerode partition)
# --------------------------------------------------------------------------


def test_minimize_equivalent_nodes():
    """A and B both have a single edge to C (same downstream structure):
    truly Myhill-Nerode equivalent, must land in the same class. C and G
    are each their own class."""
    mdp = RoutingMDP(["A", "B", "C", "G"], "G",
                     {("A", "C"): 1.0, ("B", "C"): 1.0, ("C", "G"): 1.0})
    labels = {("A", "C"): "a1", ("B", "C"): "a1", ("C", "G"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    part = minimize(dfa)
    assert part["A"] == part["B"]
    assert part["A"] != part["C"]
    assert part["C"] != part["G"]
    print("PASS minimize (equivalent nodes land in the same class)")


# --------------------------------------------------------------------------
# myhill_nerode_boundary
# --------------------------------------------------------------------------


def test_boundary_minimality_and_directedness():
    """A: a1->B, a2->D; B: a1->D; C: a1->D (single action). L(A) has an
    extra length-1 word (a2) and an extra length-2 word (a1,a1) beyond
    what C accepts; both must appear in MNB(A,C), both minimal. The
    reverse direction MNB(C,A) is empty: C accepts nothing A doesn't."""
    mdp = RoutingMDP(["A", "B", "C", "D"], "D",
                     {("A", "B"): 1.0, ("A", "D"): 1.0, ("B", "D"): 1.0,
                      ("C", "D"): 1.0})
    labels = {("A", "B"): "a1", ("A", "D"): "a2", ("B", "D"): "a1",
             ("C", "D"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    mnb_ac, mnb_ca = myhill_nerode_boundary("A", "C", dfa, 3)
    assert mnb_ac == [("a2",), ("a1", "a1")], mnb_ac
    assert mnb_ca == [], "C accepts nothing A doesn't: this direction is empty"
    print("PASS myhill_nerode_boundary (minimality, and one direction can be empty)")


def test_boundary_qq_always_empty_on_real_instance():
    """On a real seed-7 instance, q1==q2 must give an empty boundary in
    both directions: provable by construction, not just measured."""
    inst = make_pair(7, "silent_break", deterministic=True, matched=True)
    dfa = build_dfa_view(inst.m0, inst.labels)
    for q in inst.m0.nodes:
        mnb_12, mnb_21 = myhill_nerode_boundary(q, q, dfa, 3)
        assert mnb_12 == [] and mnb_21 == [], (q, mnb_12, mnb_21)
    print("PASS myhill_nerode_boundary(q, q, ...) always empty (real instance)")


def test_boundary_respects_depth_cap_on_cyclic_instance():
    """A<->B is a cycle (language would be infinite without a cap); the
    boundary in each direction must never exceed the requested length."""
    mdp = RoutingMDP(["A", "B", "C"], "C",
                     {("A", "B"): 1.0, ("B", "A"): 1.0, ("A", "C"): 1.0})
    labels = {("A", "B"): "a1", ("A", "C"): "a2", ("B", "A"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    for maxlen in (1, 2, 3, 5):
        lang = language_upto("A", dfa, maxlen)
        assert all(len(x) <= maxlen for x in lang)
        mnb_ab, mnb_ba = myhill_nerode_boundary("A", "B", dfa, maxlen)
        assert all(len(x) <= maxlen for x in mnb_ab + mnb_ba)
    assert ("a1", "a1", "a1") in language_upto("A", dfa, 3)  # A->B->A->B, still legal
    print("PASS myhill_nerode_boundary / language_upto respect the length cap on a cycle")


# --------------------------------------------------------------------------
# Fixture shared by the pair-selection tests below
# --------------------------------------------------------------------------


def _sabcg_fixture():
    """S->A->C->G and S->B->C->G: A and B are Myhill-Nerode equivalent
    (single edge each to the same C); C is reached via two genuinely
    different label sequences; A, B, G each have only one observed
    history and no alternate route (real graph invariant: only S is ever
    a starting point, so A/B/G have no OTHER path to reach them)."""
    mdp = RoutingMDP(["S", "A", "B", "C", "G"], "G",
                     {("S", "A"): 1.0, ("S", "B"): 1.0, ("A", "C"): 1.0,
                      ("B", "C"): 1.0, ("C", "G"): 1.0})
    labels = {("S", "A"): "a1", ("S", "B"): "a2", ("A", "C"): "a1",
             ("B", "C"): "a1", ("C", "G"): "a1"}
    dfa = build_dfa_view(mdp, labels)

    def _step(node, chosen, next_node, label, t, ep):
        return LiveStep(t=t, node=node, chosen=chosen, success=True,
                        next_node=next_node, phase="m0", episode_idx=ep,
                        action_label=label, parse_status="ok", retries=0,
                        raw_text="{}")

    ep0 = EpisodeOutcome(episode_idx=0, phase="m0", outcome="reached_goal",
                         steps=[_step("S", "A", "A", "a1", 1, 0),
                                _step("A", "C", "C", "a1", 2, 0),
                                _step("C", "G", "G", "a1", 3, 0)])
    ep1 = EpisodeOutcome(episode_idx=1, phase="m0", outcome="reached_goal",
                         steps=[_step("S", "B", "B", "a2", 1, 1),
                                _step("B", "C", "C", "a1", 2, 1),
                                _step("C", "G", "G", "a1", 3, 1)])
    return dfa, [ep0, ep1]


# --------------------------------------------------------------------------
# visited_nodes / visited_histories and pair selection
# --------------------------------------------------------------------------


def test_visited_nodes_and_histories():
    dfa, episodes = _sabcg_fixture()
    assert visited_nodes(episodes) == {"S", "A", "B", "C", "G"}
    hist = visited_histories(episodes)
    assert hist["A"] == [("a1",)]
    assert hist["B"] == [("a2",)]
    assert hist["C"] == [("a1", "a1"), ("a2", "a1")]
    print("PASS visited_nodes / visited_histories")


def test_sample_compression_pairs_skips():
    """C (two histories) becomes a compression pair; A, B, G each get
    exactly one observed history and no alternate route exists to them
    (nothing else points at A/B, and G is the goal), so all three are
    skipped: G specifically for being the goal, A/B for having no
    alternate history."""
    dfa, episodes = _sabcg_fixture()
    pairs, skipped = sample_compression_pairs(dfa, episodes, "S", seed=1,
                                              phase="m0", max_pairs=5)
    assert [p[0] for p in pairs] == ["C"]
    assert ("G", "goal_node") in skipped
    assert ("A", "no_alternate_history") in skipped
    assert ("B", "no_alternate_history") in skipped
    print("PASS sample_compression_pairs (goal skip, no-alternate-history skip)")


def test_sample_distinction_pairs_equivalence_skip():
    """(A, B) are Myhill-Nerode equivalent and must be skipped with that
    reason; every other visited pair is genuinely eligible."""
    dfa, episodes = _sabcg_fixture()
    pairs, skipped = sample_distinction_pairs(dfa, episodes, seed=1,
                                              phase="m0", max_pairs=10,
                                              maxlen=3)
    assert ("A", "B", "myhill_nerode_equivalent") in skipped
    got = {frozenset((q1, q2)) for q1, s1, q2, s2, m12, m21 in pairs}
    assert frozenset(("A", "B")) not in got
    assert frozenset(("A", "C")) in got and frozenset(("C", "G")) in got
    print("PASS sample_distinction_pairs (Myhill-Nerode-equivalent skip)")


def test_alternate_history_none_when_no_second_route():
    dfa, episodes = _sabcg_fixture()
    hist = visited_histories(episodes)
    assert alternate_history(dfa, "S", "A", avoid=hist["A"][0]) is None
    print("PASS alternate_history (None when no genuinely different route exists)")


def test_shortest_histories_on_seed7():
    """Sanity check against the real graph: every visited node's shortest
    history actually leads there."""
    inst = make_pair(7, "silent_break", deterministic=True, matched=True)
    dfa = build_dfa_view(inst.m0, inst.labels)
    best = shortest_histories(dfa, inst.start)
    for node, seq in best.items():
        assert run_sequence(dfa, inst.start, seq) == node
    print("PASS shortest_histories (every returned path actually reaches its node)")


# --------------------------------------------------------------------------
# sample_distinction_recall_targets: round-robin split
# --------------------------------------------------------------------------


def test_round_robin_asymmetric_split():
    """Both directions have plenty of words: at an odd total (3), the
    first-drawn direction (q1_not_q2) gets one more than the second."""
    mnb_12 = [("w1",), ("w2",), ("w3",), ("w4",)]
    mnb_21 = [("x1",), ("x2",), ("x3",)]
    rng = random.Random("roundrobin")
    drawn = sample_distinction_recall_targets(mnb_12, mnb_21, 3, rng)
    assert len(drawn) == 3
    n_a = sum(1 for _, d in drawn if d == "q1_not_q2")
    n_b = sum(1 for _, d in drawn if d == "q2_not_q1")
    assert (n_a, n_b) == (2, 1), (n_a, n_b)
    print("PASS sample_distinction_recall_targets (asymmetric split at odd total)")


def test_round_robin_one_direction_empty():
    """If one direction has no boundary words at all, every drawn word
    comes from the other: no error, no filler."""
    drawn = sample_distinction_recall_targets([("w1",), ("w2",)], [], 3,
                                              random.Random("onesided"))
    assert len(drawn) == 2  # capped by availability, not padded to 3
    assert all(d == "q1_not_q2" for _, d in drawn)
    print("PASS sample_distinction_recall_targets (one direction empty)")


def test_round_robin_no_filler_when_undersupplied():
    """Total true boundary words (2) fall short of the requested budget
    (5): exactly 2 are returned, never padded with filler."""
    drawn = sample_distinction_recall_targets([("w1",)], [("x1",)], 5,
                                              random.Random("undersupplied"))
    assert len(drawn) == 2
    print("PASS sample_distinction_recall_targets (no filler when undersupplied)")


# --------------------------------------------------------------------------
# sample_compression_targets
# --------------------------------------------------------------------------


def test_sample_compression_targets_balanced_no_duplicates():
    inst = make_pair(7, "silent_break", deterministic=True, matched=True)
    dfa = build_dfa_view(inst.m0, inst.labels)
    q = inst.start
    rng = random.Random("comptargets")
    targets = sample_compression_targets(q, dfa, 3, 4, rng)
    assert len(targets) == len(set(targets)) == 4   # no duplicates
    n_accepted = sum(1 for x in targets if is_legal_from(dfa, q, x))
    assert 1 <= n_accepted <= 3   # neither all-accepted nor all-rejected
    print("PASS sample_compression_targets (no duplicates, mixed valid/invalid)")


# --------------------------------------------------------------------------
# accepted_set
# --------------------------------------------------------------------------


def test_accepted_set_clean_violation_missing():
    candidates = {("a1",), ("a1", "a2"), ("a1", "a2", "a3")}

    clean = {("a1",): True, ("a1", "a2"): True, ("a1", "a2", "a3"): True}
    acc, viol, miss = accepted_set(candidates, clean)
    assert acc == candidates and not viol and not miss

    # prefix (a1,) answered False, but (a1,a2) itself answered True:
    # closure violation, cascades to (a1,a2,a3) too.
    broken = {("a1",): False, ("a1", "a2"): True, ("a1", "a2", "a3"): True}
    acc, viol, miss = accepted_set(candidates, broken)
    assert viol == {("a1", "a2"), ("a1", "a2", "a3")}
    assert not acc

    missing = {("a1",): True, ("a1", "a2"): None, ("a1", "a2", "a3"): True}
    acc, viol, miss = accepted_set(candidates, missing)
    assert miss == {("a1", "a2"), ("a1", "a2", "a3")}
    print("PASS accepted_set (clean / closure violation / missing answer)")


def test_accepted_set_known_violation_survives_unrelated_missing_prefix():
    """Regression: a1=False, a1a2=None, a1a2a1=True. a1a2a1 and its
    prefix a1 already prove a closure violation regardless of what the
    still-missing a1a2 turns out to be: it must not be classified as
    merely "missing"."""
    candidates = {("a1",), ("a1", "a2"), ("a1", "a2", "a1")}
    verdicts = {("a1",): False, ("a1", "a2"): None, ("a1", "a2", "a1"): True}
    acc, viol, miss = accepted_set(candidates, verdicts)
    assert ("a1", "a2", "a1") in viol
    assert ("a1", "a2", "a1") not in miss
    print("PASS accepted_set (known violation survives an unrelated missing prefix)")


# --------------------------------------------------------------------------
# score_compression: oracle + both negative-control fakes
# --------------------------------------------------------------------------


def _fake_fixture():
    """S->A(a1)->Q(a1); S->B(a2)->Q(a2); Q->G(a1), Q->H(a2). A and B have
    disjoint continuation languages (first symbol alone tells them
    apart), used to build a fake whose answer genuinely depends on which
    history it was given."""
    mdp = RoutingMDP(["S", "A", "B", "Q", "G", "H"], "G",
                     {("S", "A"): 1.0, ("S", "B"): 1.0, ("A", "Q"): 1.0,
                      ("B", "Q"): 1.0, ("Q", "G"): 1.0, ("Q", "H"): 1.0})
    labels = {("S", "A"): "a1", ("S", "B"): "a2", ("A", "Q"): "a1",
             ("B", "Q"): "a2", ("Q", "G"): "a1", ("Q", "H"): "a2"}
    return build_dfa_view(mdp, labels)


def test_score_compression_oracle_and_fakes():
    dfa = _fake_fixture()
    s1, s2 = ("a1", "a1"), ("a2", "a2")           # both reach Q
    assert run_sequence(dfa, "S", s1) == "Q"
    assert run_sequence(dfa, "S", s2) == "Q"

    rng = random.Random("compscore")
    targets = sample_compression_targets("Q", dfa, 3, 3, rng)
    cands = expand_with_prefixes(targets)

    # oracle: both evaluated from the real, true node Q -> perfectly consistent
    truth_q = {x: is_legal_from(dfa, "Q", x) for x in cands}
    assert score_compression(targets, truth_q, truth_q) == 1

    # always-start-node fake: ignores s1/s2, always evaluates from S ->
    # self-consistent (score 1) despite being flatly wrong about where it is
    from_s = {x: is_legal_from(dfa, "S", x) for x in cands}
    assert score_compression(targets, from_s, from_s) == 1

    # last-action-label fake: pretends it's standing wherever s[-1] would
    # lead FROM S, so s1 (ends "a1") and s2 (ends "a2") get genuinely
    # different, wrong answers: this is the real negative control
    def fake_last_label(s):
        pseudo = dfa_step(dfa, "S", s[-1])
        return {x: is_legal_from(dfa, pseudo, x) for x in cands}
    assert score_compression(targets, fake_last_label(s1),
                             fake_last_label(s2)) == 0
    print("PASS score_compression (oracle=1, start-node fake=1 despite being "
         "wrong, last-label fake=0)")


def test_score_compression_undefined_when_nothing_comparable():
    targets = [("a1",), ("a2",)]
    v1 = {("a1",): None, ("a2",): None}
    v2 = {("a1",): True, ("a2",): False}
    assert score_compression(targets, v1, v2) is None
    print("PASS score_compression (None when no comparable targets)")


# --------------------------------------------------------------------------
# score_distinction_recall
# --------------------------------------------------------------------------


def test_score_distinction_recall_known_values():
    """Hand-built: q1_not_q2 has two boundary words, one correctly
    recognized and one missed; q2_not_q1 was never sampled at all."""
    twd = [(("a1",), "q1_not_q2"), (("a2",), "q1_not_q2")]
    v1 = {("a1",): True, ("a2",): True}    # accepted under s1 (reaches q1)
    v2 = {("a1",): False, ("a2",): True}   # ("a2",) wrongly also accepted under s2
    rec = score_distinction_recall(twd, v1, v2)
    assert rec["q1_not_q2"]["n_evaluable"] == 2
    assert rec["q1_not_q2"]["recall"] == 0.5
    assert rec["q2_not_q1"] == {"recall": None, "n_evaluable": 0}
    print("PASS score_distinction_recall (known value + undefined-direction case)")


def test_score_distinction_recall_missing_answer_excluded():
    twd = [(("a1",), "q1_not_q2"), (("a2",), "q1_not_q2")]
    v1 = {("a1",): True, ("a2",): None}     # missing under s1
    v2 = {("a1",): False, ("a2",): False}
    rec = score_distinction_recall(twd, v1, v2)
    assert rec["q1_not_q2"]["n_evaluable"] == 1   # only ("a1",) is evaluable
    assert rec["q1_not_q2"]["recall"] == 1.0
    print("PASS score_distinction_recall (missing answer excluded from denominator)")


def test_score_distinction_recall_on_real_pair():
    """End-to-end on a real seed-7 pair: the oracle gets perfect recall
    wherever it was evaluable."""
    inst = make_pair(7, "silent_break", deterministic=True, matched=True)
    cfg = ExploreConfig(max_episodes_m0=2, max_episodes_m1=1, seed=7)
    result = run_explore_instance(
        inst, cfg,
        node_policy_fn=lambda mdp: dry_run_policy(mdp, inst.labels))
    dfa0 = build_dfa_view(inst.m0, inst.labels)
    dist_pairs, _ = sample_distinction_pairs(dfa0, result["m0_episodes"],
                                             seed=7, phase="m0",
                                             max_pairs=3, maxlen=3)
    assert dist_pairs, "expected at least one eligible distinction pair"
    rng = random.Random("realdist")
    for q1, s1, q2, s2, mnb_12, mnb_21 in dist_pairs:
        twd = sample_distinction_recall_targets(mnb_12, mnb_21, 3, rng)
        targets = [t for t, _ in twd]
        cands = expand_with_prefixes(targets)
        v1 = {x: is_legal_from(dfa0, q1, x) for x in cands}
        v2 = {x: is_legal_from(dfa0, q2, x) for x in cands}
        rec = score_distinction_recall(twd, v1, v2)
        for direction in ("q1_not_q2", "q2_not_q1"):
            if rec[direction]["n_evaluable"] > 0:
                assert rec[direction]["recall"] == 1.0, (q1, q2, direction, rec)
    print("PASS score_distinction_recall end-to-end on a real seed-7 instance (oracle)")


# --------------------------------------------------------------------------
# diagnostics_for_pair
# --------------------------------------------------------------------------


def test_diagnostics_for_pair():
    targets = [("a1", "a2")]
    truth_s1 = {("a1",): True, ("a1", "a2"): True}
    truth_s2 = {("a1",): True, ("a1", "a2"): False}
    verdicts_s1 = {("a1",): True, ("a1", "a2"): True}
    verdicts_s2 = {("a1",): False, ("a1", "a2"): False}   # wrong at length 1
    diag = diagnostics_for_pair(targets, truth_s1, truth_s2,
                                verdicts_s1, verdicts_s2)
    assert diag["first_wrong_prefix_length"][(("a1", "a2"), "s1")] is None
    assert diag["first_wrong_prefix_length"][(("a1", "a2"), "s2")] == 1
    assert diag["first_divergent_acceptance_length"][("a1", "a2")] == 1
    assert diag["accuracy_by_length"][1] == (1, 2)   # s1 correct, s2 wrong
    print("PASS diagnostics_for_pair (first-wrong-prefix / first-divergent-acceptance)")


# --------------------------------------------------------------------------
# parse_validity
# --------------------------------------------------------------------------


def test_parse_validity_status_coverage():
    assert parse_validity('{"valid": true}', ("a1",)) == \
        {"status": "ok", "valid": True}
    assert parse_validity("not json at all", ("a1",))["status"] == "malformed_json"
    assert parse_validity('{"foo": 1}', ("a1",))["status"] == "invalid_object"
    # realistic, messy model output: reasoning before the JSON, a code fence
    messy1 = ('Let me think: the action sequence a1 then a2 -- yes, that '
             'would still be a listed option here.\n{"valid": true}')
    assert parse_validity(messy1, ("a1", "a2")) == {"status": "ok", "valid": True}
    messy2 = '```json\n{"valid": false}\n```'
    assert parse_validity(messy2, ("a1",)) == {"status": "ok", "valid": False}
    print("PASS parse_validity (status coverage + realistic messy model output)")


# --------------------------------------------------------------------------
# Extra coverage: build_dfa_view / dfa_step edge cases
# --------------------------------------------------------------------------


def test_build_dfa_view_sigma_is_max_menu_size():
    """sigma is derived from the LARGEST menu anywhere in the graph, not
    any single node's own menu; a node with fewer actions still only
    lists its own labels."""
    mdp = RoutingMDP(["S", "A", "B"], "B",
                     {("S", "A"): 1.0, ("S", "B"): 1.0, ("A", "B"): 1.0})
    labels = {("S", "A"): "a1", ("S", "B"): "a2", ("A", "B"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    assert dfa.sigma == ("a1", "a2")   # S has 2 actions, the graph max
    assert dfa.menu["A"] == ["a1"]     # A itself only has 1
    print("PASS build_dfa_view (sigma == graph-wide max menu size)")


def test_build_dfa_view_sigma_survives_hard_removal_gap():
    """Regression: sigma must come from the actual label VALUES ever
    assigned, not the current max menu SIZE. C originally has 3 actions
    (a1, a2, a3); a2's edge is gone (hard-removal-style: absent from
    mdp.p, but the labels dict -- shared/stable across M0/M1 -- still
    maps it), leaving menu(C) = [a1, a3], size 2. A size-based sigma
    would only ever be ('a1', 'a2'), silently dropping a3 from every
    enumerated language even though it's still a real, legal action."""
    mdp = RoutingMDP(["S", "C", "X", "Z"], "X",
                     {("S", "C"): 1.0, ("C", "X"): 1.0, ("C", "Z"): 1.0})
    labels = {("S", "C"): "a1", ("C", "X"): "a1", ("C", "Y"): "a2",
             ("C", "Z"): "a3"}   # (C,Y)="a2" has no edge in mdp.p anymore
    dfa = build_dfa_view(mdp, labels)
    assert dfa.menu["C"] == ["a1", "a3"]
    assert "a3" in dfa.sigma
    assert dfa_step(dfa, "C", "a3") == "Z"
    assert ("a3",) in language_upto("C", dfa, 3)
    print("PASS build_dfa_view (sigma survives a hard-removal numbering gap)")


def test_dfa_step_none_node_and_unknown_label():
    """None (reject) stays absorbing when stepped again; a label outside
    a node's own menu (but within sigma) rejects without raising."""
    mdp = RoutingMDP(["S", "A"], "A", {("S", "A"): 1.0})
    labels = {("S", "A"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    assert dfa_step(dfa, None, "a1") is None
    assert dfa_step(dfa, "S", "a9") is None   # not even in sigma
    print("PASS dfa_step (None node absorbing / unknown label rejects)")


# --------------------------------------------------------------------------
# Extra coverage: language_upto exact contents
# --------------------------------------------------------------------------


def test_language_upto_exact_set_small_graph():
    """Not just a count or cap check: the exact returned set on a small,
    fully hand-verified graph."""
    mdp = RoutingMDP(["S", "A", "B"], "B",
                     {("S", "A"): 1.0, ("A", "B"): 1.0})
    labels = {("S", "A"): "a1", ("A", "B"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    assert language_upto("S", dfa, 3) == {("a1",), ("a1", "a1")}
    print("PASS language_upto (exact set on a small graph)")


# --------------------------------------------------------------------------
# Extra coverage: minimize
# --------------------------------------------------------------------------


def test_minimize_all_distinct_simple_chain():
    """A plain S->A->B chain with no symmetric structure: every node its
    own class."""
    mdp = RoutingMDP(["S", "A", "B"], "B",
                     {("S", "A"): 1.0, ("A", "B"): 1.0})
    labels = {("S", "A"): "a1", ("A", "B"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    part = minimize(dfa)
    assert len(set(part.values())) == 3
    print("PASS minimize (no symmetry: every node its own class)")


def test_minimize_transitive_chain_equivalence():
    """Two parallel 3-hop chains (A1->A2->A3->G and B1->B2->B3->G, all
    single-action nodes) are equivalent hop-for-hop, but only after the
    partition refinement iterates: A3~B3 must be found before A2~B2 can
    be, before A1~B1 can be. Checks the FIXPOINT actually iterates, not
    just a single symmetric pass."""
    mdp = RoutingMDP(["A1", "A2", "A3", "B1", "B2", "B3", "G"], "G",
                     {("A1", "A2"): 1.0, ("A2", "A3"): 1.0, ("A3", "G"): 1.0,
                      ("B1", "B2"): 1.0, ("B2", "B3"): 1.0, ("B3", "G"): 1.0})
    labels = {("A1", "A2"): "a1", ("A2", "A3"): "a1", ("A3", "G"): "a1",
             ("B1", "B2"): "a1", ("B2", "B3"): "a1", ("B3", "G"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    part = minimize(dfa)
    assert part["A1"] == part["B1"]
    assert part["A2"] == part["B2"]
    assert part["A3"] == part["B3"]
    assert len(set(part.values())) == 4   # {A1,B1}, {A2,B2}, {A3,B3}, {G}
    print("PASS minimize (transitive chain equivalence needs iteration)")


def test_minimize_different_out_degree_not_equivalent():
    """A (2 actions) can never be equivalent to C (1 action), regardless
    of downstream structure; B and C (both 1 action, same downstream)
    are equivalent to each other."""
    mdp = RoutingMDP(["A", "B", "C", "D"], "D",
                     {("A", "B"): 1.0, ("A", "D"): 1.0, ("B", "D"): 1.0,
                      ("C", "D"): 1.0})
    labels = {("A", "B"): "a1", ("A", "D"): "a2", ("B", "D"): "a1",
             ("C", "D"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    part = minimize(dfa)
    assert part["A"] != part["C"]
    assert part["B"] == part["C"]
    print("PASS minimize (different out-degree can't be equivalent)")


# --------------------------------------------------------------------------
# Extra coverage: myhill_nerode_boundary, both directions nonempty
# --------------------------------------------------------------------------


def test_boundary_both_directions_nonempty_simultaneously():
    """A: a1->X, a2->P(leaf), a3->R(leaf); has an action (a3) B lacks
    entirely. B: a1->X, a2->Q, Q: a1->Z; has a continuation past its
    a2 that A's own a2 (leading to the leaf P) doesn't. Each side has a
    genuine, independent extra capability the other lacks."""
    mdp = RoutingMDP(["A", "B", "X", "P", "R", "Q", "Z"], "X",
                     {("A", "X"): 1.0, ("A", "P"): 1.0, ("A", "R"): 1.0,
                      ("B", "X"): 1.0, ("B", "Q"): 1.0, ("Q", "Z"): 1.0})
    labels = {("A", "X"): "a1", ("A", "P"): "a2", ("A", "R"): "a3",
             ("B", "X"): "a1", ("B", "Q"): "a2", ("Q", "Z"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    mnb_ab, mnb_ba = myhill_nerode_boundary("A", "B", dfa, 3)
    assert mnb_ab == [("a3",)]
    assert mnb_ba == [("a2", "a1")]
    print("PASS myhill_nerode_boundary (both directions nonempty simultaneously)")


# --------------------------------------------------------------------------
# Extra coverage: visited_nodes / visited_histories
# --------------------------------------------------------------------------


def test_visited_skips_non_ok_steps():
    """A step whose parse_status isn't "ok" (the synthetic diagnostic
    step explore_agent.py appends when retries are exhausted) never
    counts as visiting its next_node, and never enters a history."""
    def _step(node, chosen, next_node, label, status, t, ep):
        return LiveStep(t=t, node=node, chosen=chosen, success=True,
                        next_node=next_node, phase="m0", episode_idx=ep,
                        action_label=label, parse_status=status, retries=0,
                        raw_text="{}")
    ep0 = EpisodeOutcome(episode_idx=0, phase="m0",
                         outcome="retries_exhausted", steps=[
                             _step("S", "A", "A", "a1", "ok", 1, 0),
                             _step("A", "X", "X", "a9",
                                  "retries_exhausted", 2, 0)])
    assert visited_nodes([ep0]) == {"S", "A"}
    assert "X" not in visited_histories([ep0])
    print("PASS visited_nodes / visited_histories (non-ok steps skipped)")


def test_visited_histories_on_empty_episodes():
    assert visited_nodes([]) == set()
    assert visited_histories([]) == {}
    print("PASS visited_nodes / visited_histories (empty episode list)")


def test_visited_histories_includes_start_node_with_empty_history():
    """Regression: an episode S -> A -> T never revisits S, so S must
    still appear with the empty history, or it could never be sampled
    as a compression/distinction node at all."""
    def _step(node, chosen, next_node, label, t):
        return LiveStep(t=t, node=node, chosen=chosen, success=True,
                        next_node=next_node, phase="m0", episode_idx=0,
                        action_label=label, parse_status="ok", retries=0,
                        raw_text="{}")
    ep = EpisodeOutcome(episode_idx=0, phase="m0", outcome="reached_goal",
                        steps=[_step("S", "A", "A", "a1", 1),
                               _step("A", "T", "T", "a1", 2)])
    vh = visited_histories([ep])
    assert vh["S"] == [()]
    assert "S" in visited_nodes([ep])
    print("PASS visited_histories (start node kept with the empty history)")


# --------------------------------------------------------------------------
# Extra coverage: shortest_histories / alternate_history
# --------------------------------------------------------------------------


def test_shortest_histories_unreachable_node_absent():
    mdp = RoutingMDP(["S", "A", "ISOLATED"], "A", {("S", "A"): 1.0})
    labels = {("S", "A"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    best = shortest_histories(dfa, "S")
    assert "ISOLATED" not in best
    assert alternate_history(dfa, "S", "ISOLATED", avoid=()) is None
    print("PASS shortest_histories / alternate_history (unreachable node)")


def test_alternate_history_finds_route_diverging_at_first_step():
    """S has two routes to T of equal length (via A or via B), diverging
    right away. When the plain shortest-path search's result equals
    `avoid`, alternate_history must still find the other one."""
    mdp = RoutingMDP(["S", "A", "B", "T"], "T",
                     {("S", "A"): 1.0, ("S", "B"): 1.0, ("A", "T"): 1.0,
                      ("B", "T"): 1.0})
    labels = {("S", "A"): "a1", ("S", "B"): "a2", ("A", "T"): "a1",
             ("B", "T"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    shortest = shortest_histories(dfa, "S")["T"]
    alt = alternate_history(dfa, "S", "T", avoid=shortest)
    assert alt is not None and alt != shortest
    assert run_sequence(dfa, "S", alt) == "T"
    print("PASS alternate_history (finds a route diverging at the first step)")


def test_alternate_history_finds_route_diverging_later():
    """Regression: S->A->T (shortest, avoid) and S->A->C->T both start
    with the same action; the alternate only diverges at the second
    step. A search that only ever forbids avoid's first action would
    never find this one."""
    mdp = RoutingMDP(["S", "A", "C", "T"], "T",
                     {("S", "A"): 1.0, ("A", "T"): 1.0, ("A", "C"): 1.0,
                      ("C", "T"): 1.0})
    labels = {("S", "A"): "a1", ("A", "T"): "a1", ("A", "C"): "a2",
             ("C", "T"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    shortest = shortest_histories(dfa, "S")["T"]
    assert shortest == ("a1", "a1")
    alt = alternate_history(dfa, "S", "T", avoid=shortest)
    assert alt == ("a1", "a2", "a1")
    print("PASS alternate_history (finds a route diverging after the first step)")


# --------------------------------------------------------------------------
# Extra coverage: sample_compression_pairs / sample_distinction_pairs
# --------------------------------------------------------------------------


def _four_leaf_fixture():
    """S with four independent one-hop routes to a shared goal G, each
    via a distinct intermediate node: plenty of eligible pairs for
    truncation/reproducibility tests."""
    mdp = RoutingMDP(["S", "A", "B", "C", "D", "G"], "G",
                     {("S", "A"): 1.0, ("S", "B"): 1.0, ("S", "C"): 1.0,
                      ("S", "D"): 1.0, ("A", "G"): 1.0, ("B", "G"): 1.0,
                      ("C", "G"): 1.0, ("D", "G"): 1.0})
    labels = {("S", "A"): "a1", ("S", "B"): "a2", ("S", "C"): "a3",
             ("S", "D"): "a4", ("A", "G"): "a1", ("B", "G"): "a1",
             ("C", "G"): "a1", ("D", "G"): "a1"}
    dfa = build_dfa_view(mdp, labels)

    def _step(node, chosen, next_node, label, t, ep):
        return LiveStep(t=t, node=node, chosen=chosen, success=True,
                        next_node=next_node, phase="m0", episode_idx=ep,
                        action_label=label, parse_status="ok", retries=0,
                        raw_text="{}")

    episodes = [
        EpisodeOutcome(episode_idx=i, phase="m0", outcome="reached_goal",
                       steps=[_step("S", mid, mid, lab, 1, i),
                              _step(mid, "G", "G", "a1", 2, i)])
        for i, (mid, lab) in enumerate(
            [("A", "a1"), ("B", "a2"), ("C", "a3"), ("D", "a4")])
    ]
    return dfa, episodes


def test_sample_distinction_pairs_max_pairs_truncation_and_reproducible():
    dfa, episodes = _four_leaf_fixture()
    pairs, _ = sample_distinction_pairs(dfa, episodes, seed=1, phase="m0",
                                        max_pairs=2, maxlen=3)
    assert len(pairs) == 2

    a, _ = sample_distinction_pairs(dfa, episodes, seed=1, phase="m0",
                                    max_pairs=100, maxlen=3)
    b, _ = sample_distinction_pairs(dfa, episodes, seed=1, phase="m0",
                                    max_pairs=100, maxlen=3)
    assert a == b   # same seed -> identical order and selection
    print("PASS sample_distinction_pairs (max_pairs truncation + reproducibility)")


def test_sample_compression_pairs_reproducible():
    dfa, episodes = _sabcg_fixture()
    a, _ = sample_compression_pairs(dfa, episodes, "S", seed=1, phase="m0",
                                    max_pairs=5)
    b, _ = sample_compression_pairs(dfa, episodes, "S", seed=1, phase="m0",
                                    max_pairs=5)
    assert a == b
    c, _ = sample_compression_pairs(dfa, episodes, "S", seed=1, phase="m0",
                                    max_pairs=1)
    assert len(c) == 1
    print("PASS sample_compression_pairs (reproducibility + max_pairs truncation)")


def test_sample_distinction_pairs_empty_boundary_within_cap_but_eligible():
    """A1 and B1 are genuinely Myhill-Nerode distinguishable (their paths
    diverge only at the 3rd hop: A3 continues on, B3 is a dead end), so
    minimize() correctly makes them eligible, but at maxlen=2 their
    boundary is empty in BOTH directions, since the divergence hasn't
    been reached yet. This is exactly the gap the methodology doc warns
    about between "distinguishable" and "boundary nonempty within cap"."""
    mdp = RoutingMDP(["A1", "A2", "A3", "B1", "B2", "B3", "G"], "G",
                     {("A1", "A2"): 1.0, ("A2", "A3"): 1.0, ("A3", "G"): 1.0,
                      ("B1", "B2"): 1.0, ("B2", "B3"): 1.0})
    labels = {("A1", "A2"): "a1", ("A2", "A3"): "a1", ("A3", "G"): "a1",
             ("B1", "B2"): "a1", ("B2", "B3"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    part = minimize(dfa)
    assert part["A1"] != part["B1"]   # genuinely distinguishable

    mnb_12, mnb_21 = myhill_nerode_boundary("A1", "B1", dfa, 2)
    assert mnb_12 == [] and mnb_21 == []   # but empty within a short cap

    # A1 and B1 are each an episode's own starting node (visited with the
    # empty history), so the pair is actually sampled.
    def _step(node, chosen, next_node, label, t, ep):
        return LiveStep(t=t, node=node, chosen=chosen, success=True,
                        next_node=next_node, phase="m0", episode_idx=ep,
                        action_label=label, parse_status="ok", retries=0,
                        raw_text="{}")
    episodes = [
        EpisodeOutcome(episode_idx=0, phase="m0", outcome="horizon_cutoff",
                       steps=[_step("A1", "A2", "A2", "a1", 1, 0)]),
        EpisodeOutcome(episode_idx=1, phase="m0", outcome="horizon_cutoff",
                       steps=[_step("B1", "B2", "B2", "a1", 1, 1)]),
    ]
    pairs, _ = sample_distinction_pairs(dfa, episodes, seed=1, phase="m0",
                                        max_pairs=100, maxlen=2)
    target = next(p for p in pairs if {p[0], p[2]} == {"A1", "B1"})
    _, _, _, _, got_12, got_21 = target
    assert got_12 == [] and got_21 == []
    print("PASS sample_distinction_pairs (eligible via minimize(), empty "
         "boundary within the length cap)")


# --------------------------------------------------------------------------
# Extra coverage: sample_compression_targets
# --------------------------------------------------------------------------


def test_sample_compression_targets_n_exceeds_universe_and_zero():
    mdp = RoutingMDP(["S", "A"], "A", {("S", "A"): 1.0})
    labels = {("S", "A"): "a1"}
    dfa = build_dfa_view(mdp, labels)   # sigma=('a1',): universe at maxlen=3 has exactly 3 strings
    t = sample_compression_targets("S", dfa, 3, 100, random.Random("x"))
    assert len(t) == 3 == len(set(t))   # capped at the universe size, no duplicates
    assert sample_compression_targets("S", dfa, 3, 0, random.Random("x")) == []
    print("PASS sample_compression_targets (n > universe size, n == 0)")


def test_sample_compression_targets_reproducible():
    dfa, _ = _four_leaf_fixture()
    a = sample_compression_targets("S", dfa, 3, 3, random.Random("seedx"))
    b = sample_compression_targets("S", dfa, 3, 3, random.Random("seedx"))
    assert a == b
    print("PASS sample_compression_targets (reproducible with the same rng seed)")


# --------------------------------------------------------------------------
# Extra coverage: sample_distinction_recall_targets
# --------------------------------------------------------------------------


def test_round_robin_both_directions_empty():
    assert sample_distinction_recall_targets([], [], 3,
                                             random.Random("bothempty")) == []
    print("PASS sample_distinction_recall_targets (both directions empty)")


def test_round_robin_exact_n_total_when_plenty_available():
    mnb_12 = [(f"w{i}",) for i in range(5)]
    mnb_21 = [(f"x{i}",) for i in range(5)]
    drawn = sample_distinction_recall_targets(mnb_12, mnb_21, 4,
                                              random.Random("plenty"))
    assert len(drawn) == 4
    print("PASS sample_distinction_recall_targets (exact budget met when plenty available)")


# --------------------------------------------------------------------------
# Extra coverage: expand_with_prefixes / true_labels
# --------------------------------------------------------------------------


def test_expand_with_prefixes_dedup_shared_prefix():
    out = expand_with_prefixes([("a1", "a2"), ("a1", "a3")])
    assert out == {("a1",), ("a1", "a2"), ("a1", "a3")}
    print("PASS expand_with_prefixes (shared prefix deduplicated)")


def test_true_labels_matches_is_legal_from():
    mdp = RoutingMDP(["S", "A"], "A", {("S", "A"): 1.0})
    labels = {("S", "A"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    candidates = {("a1",), ("a2",)}
    assert true_labels(candidates, dfa, "S") == {("a1",): True, ("a2",): False}
    print("PASS true_labels (matches is_legal_from exactly)")


# --------------------------------------------------------------------------
# Extra coverage: accepted_set
# --------------------------------------------------------------------------


def test_accepted_set_length1_has_no_prefixes_to_violate():
    acc, viol, miss = accepted_set({("a1",)}, {("a1",): True})
    assert acc == {("a1",)} and not viol and not miss
    print("PASS accepted_set (length-1 candidate: no prefixes, trivially clean)")


def test_accepted_set_rejection_is_not_a_violation():
    """own=False is a plain rejection, never a closure violation:
    violations are specifically own=True with a False prefix."""
    acc, viol, miss = accepted_set({("a1",), ("a1", "a2")},
                                   {("a1",): False, ("a1", "a2"): False})
    assert not viol
    assert not acc
    print("PASS accepted_set (a plain rejection is never a closure violation)")


# --------------------------------------------------------------------------
# Extra coverage: score_compression / score_distinction_recall
# --------------------------------------------------------------------------


def test_score_compression_partial_missing_still_scores():
    targets = [("a1",), ("a2",)]
    v1 = {("a1",): None, ("a2",): True}
    v2 = {("a1",): True, ("a2",): True}
    assert score_compression(targets, v1, v2) == 1   # only (a2,) is comparable, agrees
    print("PASS score_compression (scores over the comparable subset)")


def test_score_compression_agreeing_on_rejection_scores_one():
    targets = [("a1",), ("a2",)]
    v1 = {("a1",): False, ("a2",): False}
    v2 = {("a1",): False, ("a2",): False}
    assert score_compression(targets, v1, v2) == 1
    print("PASS score_compression (agreement doesn't require acceptance, just consistency)")


def test_score_compression_catches_prefix_level_divergence():
    """Regression: target (a1,a2); both sides agree it's itself invalid,
    but disagree on its prefix a1 (valid under s1, invalid under s2) --
    which was queried too. That divergence is real evidence of a
    compression error and must not be hidden just because it showed up
    on a prefix instead of the target."""
    targets = [("a1", "a2")]
    v1 = {("a1",): True, ("a1", "a2"): False}
    v2 = {("a1",): False, ("a1", "a2"): False}
    assert score_compression(targets, v1, v2) == 0
    print("PASS score_compression (catches divergence on a queried prefix)")


def test_score_distinction_recall_both_directions_nontrivial_together():
    twd = [(("a1",), "q1_not_q2"), (("a2",), "q2_not_q1")]
    v1 = {("a1",): True, ("a2",): True}     # s1: a1 correctly seen valid; a2 wrongly also seen valid
    v2 = {("a1",): False, ("a2",): False}   # s2: correctly rejects both
    rec = score_distinction_recall(twd, v1, v2)
    assert rec["q1_not_q2"]["recall"] == 1.0
    assert rec["q2_not_q1"]["recall"] == 0.0
    print("PASS score_distinction_recall (both directions scored independently, together)")


# --------------------------------------------------------------------------
# Extra coverage: diagnostics_for_pair
# --------------------------------------------------------------------------


def test_diagnostics_accuracy_by_length_and_closure_violations():
    targets = [("a1", "a2")]
    truth_s1 = {("a1",): True, ("a1", "a2"): True}
    truth_s2 = {("a1",): True, ("a1", "a2"): True}
    v1 = {("a1",): False, ("a1", "a2"): True}   # closure violation for s1
    v2 = {("a1",): True, ("a1", "a2"): False}
    diag = diagnostics_for_pair(targets, truth_s1, truth_s2, v1, v2)
    assert ("a1", "a2") in diag["closure_violations"]["s1"]
    assert not diag["closure_violations"]["s2"]
    assert diag["accuracy_by_length"][1] == (1, 2)   # s2 right at len1, s1 wrong
    print("PASS diagnostics_for_pair (accuracy_by_length + closure_violations populated)")


def test_diagnostics_missing_guard_suppresses_spurious_divergence():
    """s1 is missing at length 1 (so accepted_set cascades that
    missingness through the whole candidate); s2 is fully, cleanly
    answered. Without checking the missing sets, a naive membership
    comparison would misread this as a divergence purely from missing
    data on one side: the guard must report None instead."""
    targets = [("a1", "a2")]
    truth_s1 = {("a1",): True, ("a1", "a2"): True}
    truth_s2 = {("a1",): True, ("a1", "a2"): True}
    v1 = {("a1",): None, ("a1", "a2"): True}
    v2 = {("a1",): True, ("a1", "a2"): True}
    diag = diagnostics_for_pair(targets, truth_s1, truth_s2, v1, v2)
    assert diag["first_divergent_acceptance_length"][("a1", "a2")] is None
    print("PASS diagnostics_for_pair (missing data never reported as a spurious divergence)")


def test_diagnostics_never_diverges_is_none():
    targets = [("a1", "a2", "a3")]
    truth = {("a1",): True, ("a1", "a2"): True, ("a1", "a2", "a3"): True}
    v = dict(truth)
    diag = diagnostics_for_pair(targets, truth, truth, v, dict(v))
    assert diag["first_divergent_acceptance_length"][("a1", "a2", "a3")] is None
    print("PASS diagnostics_for_pair (identical answers never register a divergence)")


# --------------------------------------------------------------------------
# Extra coverage: parse_validity / dry-run oracle
# --------------------------------------------------------------------------


def test_parse_validity_non_bool_empty_and_extra_fields():
    assert parse_validity('{"valid": "true"}', ("a1",))["status"] == "invalid_object"
    assert parse_validity("", ("a1",))["status"] == "malformed_json"
    assert parse_validity('{"valid": true, "extra": 1}', ("a1",)) == \
        {"status": "ok", "valid": True}
    print("PASS parse_validity (non-bool value, empty input, ignored extra fields)")


def test_dry_run_legality_reply_true_and_false():
    mdp = RoutingMDP(["S", "A"], "A", {("S", "A"): 1.0})
    labels = {("S", "A"): "a1"}
    dfa = build_dfa_view(mdp, labels)
    assert dry_run_legality_reply(dfa, "S", ("a1",)) == '{"valid": true}'
    assert dry_run_legality_reply(dfa, "S", ("a2",)) == '{"valid": false}'
    print("PASS dry_run_legality_reply (true/false round-trip)")


# --------------------------------------------------------------------------
# Property checks across many real generated instances (not hand-built
# fixtures): catches anything a curated set of examples could miss by
# chance, at the cost of testing "the code agrees with a differently
# structured reference computation" rather than a hand-verified value.
# --------------------------------------------------------------------------


def _brute_force_boundary(q1, q2, dfa, maxlen):
    """Independent reference for myhill_nerode_boundary: enumerate every
    string up to maxlen via itertools.product (not
    coherence._sigma_strings_upto's incremental frontier-building), check
    membership in L(q1)\\L(q2) one candidate at a time (not via a set
    difference), then filter to minimal words. Shares only is_legal_from
    with the implementation under test; everything else is a
    differently structured computation of the same definition."""
    diff = set()
    for length in range(1, maxlen + 1):
        for combo in product(dfa.sigma, repeat=length):
            if is_legal_from(dfa, q1, combo) and not is_legal_from(dfa, q2, combo):
                diff.add(combo)
    out = [x for x in diff
          if not any(x[:j] in diff for j in range(1, len(x)))]
    out.sort(key=lambda x: (len(x), x))
    return out


def test_boundary_matches_independent_brute_force_across_seeds():
    """Not a single hand-built example: over 60 real generated instances,
    every word myhill_nerode_boundary returns for every node pair must
    match an independently computed reference exactly, in both
    directions. This is what actually caught a real bug during
    development: without a lexicographic tiebreak, same-length words came
    back in an order that depended on incidental Python set-iteration
    layout, which no single hand-built fixture happened to expose (see
    myhill_nerode_boundary's docstring). Covers both no_change (nothing
    ever removed) and hard_removal (labels can have numbering gaps,
    checked on both M0 and M1) -- no_change alone would never exercise
    the gap case a size-based sigma got wrong (see
    test_build_dfa_view_sigma_survives_hard_removal_gap)."""
    maxlen = 3
    n_pairs_checked = 0
    for condition in ("no_change", "hard_removal"):
        for seed in range(60):
            inst = make_pair(seed, condition, deterministic=True)
            for mdp in (inst.m0, inst.m1):
                dfa = build_dfa_view(mdp, inst.labels)
                nodes = mdp.nodes
                for i, q1 in enumerate(nodes):
                    for q2 in nodes[i + 1:]:
                        got_12, got_21 = myhill_nerode_boundary(q1, q2, dfa, maxlen)
                        want_12 = _brute_force_boundary(q1, q2, dfa, maxlen)
                        want_21 = _brute_force_boundary(q2, q1, dfa, maxlen)
                        assert got_12 == want_12, (condition, seed, q1, q2, got_12, want_12)
                        assert got_21 == want_21, (condition, seed, q1, q2, got_21, want_21)
                        n_pairs_checked += 1
    assert n_pairs_checked > 5000, n_pairs_checked
    print(f"PASS myhill_nerode_boundary matches independent brute force "
         f"across {n_pairs_checked} (condition, seed, node-pair, direction) checks")


def test_minimize_same_class_implies_identical_language_across_seeds():
    """Property check complementing the hand-built equivalence fixture:
    over 60 real generated instances, wherever minimize() places two or
    more nodes in the same class, their accepted languages (up to a
    generous length) must be identical: a necessary condition for true
    Myhill-Nerode equivalence that doesn't depend on any single curated
    example. Multi-member classes are naturally rare in these graphs
    (matches the ~0.17-per-instance rate measured in the methodology
    doc), so this also reports how many it actually found across the 60
    seeds, not just that zero would have trivially passed. Covers both
    no_change and hard_removal (M0 and M1), for the same reason as
    test_boundary_matches_independent_brute_force_across_seeds."""
    maxlen = 4
    n_classes_checked = 0
    for condition in ("no_change", "hard_removal"):
        for seed in range(60):
            inst = make_pair(seed, condition, deterministic=True)
            for mdp in (inst.m0, inst.m1):
                dfa = build_dfa_view(mdp, inst.labels)
                part = minimize(dfa)
                by_class = {}
                for node, cls in part.items():
                    by_class.setdefault(cls, []).append(node)
                for members in by_class.values():
                    if len(members) < 2:
                        continue
                    langs = [language_upto(m, dfa, maxlen) for m in members]
                    assert all(l == langs[0] for l in langs), \
                        (condition, seed, members, langs)
                    n_classes_checked += 1
    print(f"PASS minimize (same-class nodes have identical language across "
         f"{n_classes_checked} multi-member classes found over 120 "
         f"(condition, seed, world) combinations)")


# --------------------------------------------------------------------------
# End-to-end dry run
# --------------------------------------------------------------------------


def test_end_to_end_dry_run_seed7():
    """Full pipeline on a real seed-7 instance: sample pairs, sample
    targets, answer via the oracle, score. Compression must be 1
    everywhere evaluable; distinction recall must be 1.0 everywhere
    evaluable."""
    inst = make_pair(7, "silent_break", deterministic=True, matched=True)
    cfg = ExploreConfig(max_episodes_m0=2, max_episodes_m1=1, seed=7)
    result = run_explore_instance(
        inst, cfg,
        node_policy_fn=lambda mdp: dry_run_policy(mdp, inst.labels))
    dfa0 = build_dfa_view(inst.m0, inst.labels)
    rng = random.Random("e2e")

    comp_pairs, _ = sample_compression_pairs(dfa0, result["m0_episodes"],
                                             inst.start, seed=7, phase="m0",
                                             max_pairs=4)
    assert comp_pairs
    for q, s1, s2 in comp_pairs:
        targets = sample_compression_targets(q, dfa0, 3, 3, rng)
        cands = expand_with_prefixes(targets)
        truth = {x: is_legal_from(dfa0, q, x) for x in cands}
        assert score_compression(targets, truth, truth) == 1

    dist_pairs, _ = sample_distinction_pairs(dfa0, result["m0_episodes"],
                                             seed=7, phase="m0",
                                             max_pairs=4, maxlen=3)
    assert dist_pairs
    for q1, s1, q2, s2, mnb_12, mnb_21 in dist_pairs:
        twd = sample_distinction_recall_targets(mnb_12, mnb_21, 3, rng)
        targets = [t for t, _ in twd]
        cands = expand_with_prefixes(targets)
        v1 = {x: is_legal_from(dfa0, q1, x) for x in cands}
        v2 = {x: is_legal_from(dfa0, q2, x) for x in cands}
        rec = score_distinction_recall(twd, v1, v2)
        for direction in ("q1_not_q2", "q2_not_q1"):
            if rec[direction]["n_evaluable"] > 0:
                assert rec[direction]["recall"] == 1.0

    # dry_run_legality_reply / make_dry_run_answer_fn round-trip
    answer_fn = make_dry_run_answer_fn(dfa0, inst.start)
    raw, reasoning, usage = answer_fn([], ("a1",))
    assert parse_validity(raw, ("a1",))["status"] == "ok"
    assert dry_run_legality_reply(dfa0, inst.start, ("a1",)) == raw
    print("PASS end-to-end dry run on a real seed-7 instance (oracle throughout)")


if __name__ == "__main__":
    test_dfa_step_goal_silent_break_hard_removal()
    test_run_sequence_reject_is_absorbing()
    test_minimize_equivalent_nodes()
    test_boundary_minimality_and_directedness()
    test_boundary_qq_always_empty_on_real_instance()
    test_boundary_respects_depth_cap_on_cyclic_instance()
    test_visited_nodes_and_histories()
    test_sample_compression_pairs_skips()
    test_sample_distinction_pairs_equivalence_skip()
    test_alternate_history_none_when_no_second_route()
    test_shortest_histories_on_seed7()
    test_round_robin_asymmetric_split()
    test_round_robin_one_direction_empty()
    test_round_robin_no_filler_when_undersupplied()
    test_sample_compression_targets_balanced_no_duplicates()
    test_accepted_set_clean_violation_missing()
    test_accepted_set_known_violation_survives_unrelated_missing_prefix()
    test_score_compression_oracle_and_fakes()
    test_score_compression_undefined_when_nothing_comparable()
    test_score_distinction_recall_known_values()
    test_score_distinction_recall_missing_answer_excluded()
    test_score_distinction_recall_on_real_pair()
    test_diagnostics_for_pair()
    test_parse_validity_status_coverage()
    test_build_dfa_view_sigma_is_max_menu_size()
    test_build_dfa_view_sigma_survives_hard_removal_gap()
    test_dfa_step_none_node_and_unknown_label()
    test_language_upto_exact_set_small_graph()
    test_minimize_all_distinct_simple_chain()
    test_minimize_transitive_chain_equivalence()
    test_minimize_different_out_degree_not_equivalent()
    test_boundary_both_directions_nonempty_simultaneously()
    test_visited_skips_non_ok_steps()
    test_visited_histories_on_empty_episodes()
    test_visited_histories_includes_start_node_with_empty_history()
    test_shortest_histories_unreachable_node_absent()
    test_alternate_history_finds_route_diverging_at_first_step()
    test_alternate_history_finds_route_diverging_later()
    test_sample_distinction_pairs_max_pairs_truncation_and_reproducible()
    test_sample_compression_pairs_reproducible()
    test_sample_distinction_pairs_empty_boundary_within_cap_but_eligible()
    test_sample_compression_targets_n_exceeds_universe_and_zero()
    test_sample_compression_targets_reproducible()
    test_round_robin_both_directions_empty()
    test_round_robin_exact_n_total_when_plenty_available()
    test_expand_with_prefixes_dedup_shared_prefix()
    test_true_labels_matches_is_legal_from()
    test_accepted_set_length1_has_no_prefixes_to_violate()
    test_accepted_set_rejection_is_not_a_violation()
    test_score_compression_partial_missing_still_scores()
    test_score_compression_agreeing_on_rejection_scores_one()
    test_score_compression_catches_prefix_level_divergence()
    test_score_distinction_recall_both_directions_nontrivial_together()
    test_diagnostics_accuracy_by_length_and_closure_violations()
    test_diagnostics_missing_guard_suppresses_spurious_divergence()
    test_diagnostics_never_diverges_is_none()
    test_parse_validity_non_bool_empty_and_extra_fields()
    test_dry_run_legality_reply_true_and_false()
    test_boundary_matches_independent_brute_force_across_seeds()
    test_minimize_same_class_implies_identical_language_across_seeds()
    test_end_to_end_dry_run_seed7()
    print("\nALL COHERENCE TESTS PASSED")
