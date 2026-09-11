#!/usr/bin/env python3
"""World-model coherence probes (compression + distinction-recall) for the
active-exploration pilot.

Adapts the DFA / Myhill-Nerode framework from Vafa et al. 2024
("Evaluating the World Model Implicit in a Generative Model", arXiv
2406.03689) to a deterministic RoutingMDP; see docs/world_model_coherence_active.txt
for the formal definitions and docs/coherence_probes_implementation_plan.txt
for the design decisions this module implements (v1 scope: compression and
distinction-recall only, distinction-precision deferred).

Everything in this module is offline and network-free: it computes DFA
ground truth, samples candidates, and scores model answers that are
supplied to it. The model-calling loop that actually elicits those answers
lives in run_pilot.py (run_coherence_probes), matching how explore_agent.py
and explore_metrics.py divide the same responsibility for the other probes.

Stdlib only. Python 3.8+.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from explore_agent import extract_last_json_object
from resource_mdp import invert_labels, legal_actions

# --------------------------------------------------------------------------
# DFA view and one-step transition
# --------------------------------------------------------------------------


@dataclass
class DFAView:
    """Bundles a RoutingMDP with its label menu and label->destination
    map, so callers don't have to re-derive them at every call site.

    mdp: the RoutingMDP (M0 or M1) this view is over.
    menu: {node: ['a1', 'a2', ...]} from resource_mdp.legal_actions;
        absent for a node with no outgoing edges (the goal).
    inv: {(node, 'aK'): destination} from resource_mdp.invert_labels.
    sigma: the full label alphabet ('a1', ..., 'aK'), K = the largest
        menu anywhere in the graph; not every label is legal at every
        node, exactly as in the underlying DFA's delta.
    """
    mdp: object
    menu: dict
    inv: dict
    sigma: tuple


def build_dfa_view(mdp, labels) -> DFAView:
    menu = legal_actions(mdp, labels)
    inv = invert_labels(labels)
    # sigma must be every label VALUE ever assigned in `labels` (shared,
    # stable across M0/M1), not derived from the current mdp's max menu
    # SIZE: under hard_removal a menu shrinks but keeps its surviving
    # labels' original numbers, e.g. a 2-action menu can be {a2, a3}.
    # Deriving sigma from size alone would then never generate 'a3' as a
    # candidate at all, silently dropping it from every enumerated
    # language even though dfa_step (via the real per-node menu) accepts
    # it fine.
    sigma = tuple(sorted({lab for lab in labels.values()},
                        key=lambda s: int(s[1:])))
    return DFAView(mdp=mdp, menu=menu, inv=inv, sigma=sigma)


def dfa_step(dfa, node, label):
    """One DFA transition. None = reject (label not in node's current
    menu; also correctly absorbing when `node` is itself None). If the
    label is legal but the edge is a silent break (probability 0.0 in the
    live mdp.p), the step self-loops, returning `node` itself rather than
    the nominal destination: the tau_M(q,a)=q rule from the design doc."""
    if label not in dfa.menu.get(node, ()):
        return None
    dest = dfa.inv[(node, label)]
    if dfa.mdp.p.get((node, dest)) == 0.0:
        return node
    return dest


def run_sequence(dfa, start, labels_seq):
    """Fold dfa_step over a label sequence. Returns the node reached, or
    None if any step rejects along the way (reject is absorbing)."""
    node = start
    for lab in labels_seq:
        if node is None:
            return None
        node = dfa_step(dfa, node, lab)
    return node


def is_legal_from(dfa, start, labels_seq):
    """True iff every step of labels_seq is legal starting from `start`:
    x in L^W(start) membership, per the methodology doc. True for the
    empty sequence (vacuous; L^W itself is only defined over Sigma^+, so
    callers never query the empty sequence directly)."""
    return run_sequence(dfa, start, labels_seq) is not None


# --------------------------------------------------------------------------
# Ground truth: exhaustive and offline (no model calls anywhere below)
# --------------------------------------------------------------------------


def _sigma_strings_upto(sigma, maxlen):
    """Every non-empty string over sigma with length <= maxlen, as label
    tuples, shortest first."""
    out = []
    frontier = [()]
    for _ in range(maxlen):
        nxt = []
        for s in frontier:
            for a in sigma:
                x = s + (a,)
                out.append(x)
                nxt.append(x)
        frontier = nxt
    return out


def language_upto(q, dfa, maxlen):
    """{x in Sigma^{<=maxlen} : is_legal_from(dfa, q, x)}: exhaustive but
    purely offline (no API calls). The ground-truth basis for both
    probes; unaffected by how many candidates later get sampled for
    querying the model."""
    return {x for x in _sigma_strings_upto(dfa.sigma, maxlen)
            if is_legal_from(dfa, q, x)}


def minimize(dfa):
    """Coarsest state partition of the DFA (Moore refinement, exact, no
    length cap): {node: class_id}. Two nodes in the same class are
    truly Myhill-Nerode equivalent (never distinguishable at any
    length). Used to decide distinction-pair eligibility: a
    length-bounded empty myhill_nerode_boundary is not sufficient for
    that, since a pair's shortest distinguishing word can exceed the
    candidate length cap used for sampling (measured: at length 3, 76%
    of ordered pairs already show a non-empty boundary, vs. an average
    7.85/8 truly distinguishable classes via this exact refinement; the
    gap is pairs whose shortest distinguisher is longer than 3)."""
    nodes = list(dfa.mdp.nodes)
    part = {q: 0 for q in nodes}
    while True:
        sig = {q: (part[q], tuple(part.get(dfa_step(dfa, q, a), -1)
                                  for a in dfa.sigma))
               for q in nodes}
        idx, new_part = {}, {}
        for q in nodes:
            key = sig[q]
            if key not in idx:
                idx[key] = len(idx)
            new_part[q] = idx[key]
        if new_part == part:
            return part
        part = new_part


def myhill_nerode_boundary(q1, q2, dfa, maxlen):
    """Minimal distinguishing words in each direction, up to length
    `maxlen`: a direct filter over language_upto(q1)/language_upto(q2),
    not a product-automaton search, so there is no state-deduplication
    question to get wrong (see docs/coherence_probes_implementation_plan.txt).

    x in L(q1) implies every proper prefix of x is also in L(q1) (reject
    is absorbing along the walk), so for x in L(q1)\\L(q2), the paper's
    minimality condition ("every proper prefix in the interior
    L(q1) ∩ L(q2)") reduces to: no proper prefix of x is itself in
    L(q1)\\L(q2). That's what is checked directly below.

    At q1 == q2, language_upto(q1) == language_upto(q2), so both
    returned lists are empty by construction, no special-casing needed
    (see test_coherence.py for the direct check).

    Returns (mnb_12, mnb_21): lists of label tuples sorted by (length,
    lexicographic value), a full, deterministic order, not just "by
    length": iterating a Python set of tuples has no defined order, so
    without the lexicographic tiebreak, two calls with the same true
    boundary can return the same words in a different order depending on
    the set's internal layout (found by a randomized property test across
    many real instances, docs/coherence_probes_implementation_plan.txt
    Part 4). Harmless downstream (sample_distinction_recall_targets
    shuffles its input anyway), but the function's own contract should
    not depend on incidental set-iteration order.
    """
    l1 = language_upto(q1, dfa, maxlen)
    l2 = language_upto(q2, dfa, maxlen)

    def _minimal_diff(a, b):
        diff = a - b
        out = [x for x in diff
              if not any(x[:j] in diff for j in range(1, len(x)))]
        out.sort(key=lambda x: (len(x), x))
        return out

    return _minimal_diff(l1, l2), _minimal_diff(l2, l1)


# --------------------------------------------------------------------------
# Visited nodes / histories, from explore_agent's episode logs
# --------------------------------------------------------------------------


def visited_nodes(episodes):
    """Every node visited: each episode's own starting node
    (ep.steps[0].node, regardless of parse_status), plus LiveStep.node/
    .next_node for every step that parses "ok"."""
    seen = set()
    for ep in episodes:
        if ep.steps:
            seen.add(ep.steps[0].node)
        for st in ep.steps:
            if st.parse_status != "ok":
                continue
            seen.add(st.node)
            seen.add(st.next_node)
    return seen


def visited_histories(episodes):
    """{node: [label_seq, ...]}: the distinct real action-label
    sequences that reached each node, in first-seen order. Includes each
    episode's starting node with the empty sequence, so a start node
    with no return loop can still be sampled. Steps whose parse_status
    isn't "ok" are skipped for the action-label histories."""
    out = {}
    for ep in episodes:
        if ep.steps:
            bucket = out.setdefault(ep.steps[0].node, [])
            if () not in bucket:
                bucket.append(())
        hist = []
        for st in ep.steps:
            if st.parse_status != "ok":
                continue
            hist.append(st.action_label)
            seq = tuple(hist)
            bucket = out.setdefault(st.next_node, [])
            if seq not in bucket:
                bucket.append(seq)
    return out


# --------------------------------------------------------------------------
# Constructed (counterfactual) histories, for compression's second history
# --------------------------------------------------------------------------


def shortest_histories(dfa, start):
    """{node: shortest legal label-sequence from start}, for every
    reachable node. State space is finite, so this always terminates."""
    best = {start: ()}
    frontier = [start]
    while frontier:
        nxt = []
        for node in frontier:
            for a in dfa.sigma:
                v = dfa_step(dfa, node, a)
                if v is not None and v not in best:
                    best[v] = best[node] + (a,)
                    nxt.append(v)
        frontier = nxt
    return best


def alternate_history(dfa, start, target, avoid):
    """A second, legal label-sequence from `start` to `target`, distinct
    from `avoid` anywhere along the way, not just at the first step.
    Searches over (node, diverged) instead of just node, where diverged
    means the path so far already differs from avoid's own prefix of the
    same length; once true, any continuation qualifies. Bounds the
    search to about n_nodes + len(avoid) states. None if no such path
    exists (the pair is then skipped by the caller)."""
    shortest = shortest_histories(dfa, start).get(target)
    if shortest is None:
        return None
    if shortest != avoid:
        return shortest
    avoid_len = len(avoid)
    best = {(start, False): ()}
    frontier = [(start, False)]
    while frontier:
        nxt = []
        for node, diverged in frontier:
            path = best[(node, diverged)]
            for a in dfa.sigma:
                v = dfa_step(dfa, node, a)
                if v is None:
                    continue
                p2 = path + (a,)
                d2 = (diverged or len(p2) > avoid_len
                     or p2[-1] != avoid[len(p2) - 1])
                if v == target and d2:
                    return p2
                key = (v, d2)
                if key not in best:
                    best[key] = p2
                    nxt.append(key)
        frontier = nxt
    return None


# --------------------------------------------------------------------------
# Pair selection: visited nodes only, deterministic, seeded
# --------------------------------------------------------------------------


def sample_compression_pairs(dfa, episodes, start, seed, phase, max_pairs):
    """Up to max_pairs (q, s1, s2) compression candidates: distinct
    histories reaching the same visited, non-goal node (the goal has
    L^W(goal) = empty and cannot serve as a compression anchor, see the
    methodology doc). If a node has only one observed history, a second
    is constructed via alternate_history; nodes for which no second
    history exists (observed or constructed) are skipped.

    Returns (candidates, skipped); skipped is a list of (node, reason).
    """
    rng = random.Random(f"pilot|{seed}|compression|{phase}")
    hist = visited_histories(episodes)
    goal = dfa.mdp.goal
    candidates, skipped = [], []
    for q in sorted(hist):
        if q == goal:
            skipped.append((q, "goal_node"))
            continue
        seqs = hist[q]
        if len(seqs) >= 2:
            s1, s2 = rng.sample(seqs, 2)
        else:
            s1 = seqs[0]
            s2 = alternate_history(dfa, start, q, avoid=s1)
            if s2 is None:
                skipped.append((q, "no_alternate_history"))
                continue
        candidates.append((q, s1, s2))
    rng.shuffle(candidates)
    return candidates[:max_pairs], skipped


def sample_distinction_pairs(dfa, episodes, seed, phase, max_pairs, maxlen):
    """Up to max_pairs (q1, s1, q2, s2, mnb_12, mnb_21) distinction
    candidates: pairs of visited, genuinely Myhill-Nerode-distinguishable
    nodes (per minimize(), not a length-bounded boundary check; see
    myhill_nerode_boundary's docstring for why), each carrying one
    observed history to each node.

    Returns (candidates, skipped); skipped is a list of
    (q1, q2, reason).
    """
    rng = random.Random(f"pilot|{seed}|distinction|{phase}")
    hist = visited_histories(episodes)
    part = minimize(dfa)
    nodes = sorted(hist)
    candidates, skipped = [], []
    for i, q1 in enumerate(nodes):
        for q2 in nodes[i + 1:]:
            if part[q1] == part[q2]:
                skipped.append((q1, q2, "myhill_nerode_equivalent"))
                continue
            mnb_12, mnb_21 = myhill_nerode_boundary(q1, q2, dfa, maxlen)
            s1 = rng.choice(hist[q1])
            s2 = rng.choice(hist[q2])
            candidates.append((q1, s1, q2, s2, mnb_12, mnb_21))
    rng.shuffle(candidates)
    return candidates[:max_pairs], skipped


# --------------------------------------------------------------------------
# Target (candidate) sampling: what actually gets put to the model
# --------------------------------------------------------------------------


def sample_compression_targets(q, dfa, maxlen, n, rng):
    """n candidates (default 3), sampled without replacement, balanced
    between accepted and rejected strings in Sigma^{<=maxlen} at q: the
    model has genuine opportunity to be wrong on either side, which
    is what gives compression's score any teeth (see the methodology
    doc's "compression as a special case of distinction")."""
    universe = _sigma_strings_upto(dfa.sigma, maxlen)
    accepted = [x for x in universe if is_legal_from(dfa, q, x)]
    accepted_ids = set(accepted)
    rejected = [x for x in universe if x not in accepted_ids]
    rng.shuffle(accepted)
    rng.shuffle(rejected)
    out, ai, ri, want_accepted = [], 0, 0, True
    while len(out) < n and (ai < len(accepted) or ri < len(rejected)):
        if want_accepted and ai < len(accepted):
            out.append(accepted[ai]); ai += 1
        elif ri < len(rejected):
            out.append(rejected[ri]); ri += 1
        elif ai < len(accepted):
            out.append(accepted[ai]); ai += 1
        want_accepted = not want_accepted
    return out


def sample_distinction_recall_targets(mnb_12, mnb_21, n_total, rng):
    """Strict round-robin over the two independently shuffled boundary
    word lists, starting with the q1-vs-q2 direction, until n_total
    words are drawn or both lists are exhausted. No filler: if fewer
    than n_total true boundary words exist in total, fewer are sampled,
    never padded. At odd n_total, the first-drawn direction gets one
    more word than the second, a fixed consequence of the round-robin
    order, not a random one.

    Returns a list of (word, direction), direction in
    {"q1_not_q2", "q2_not_q1"}.
    """
    list_a, list_b = list(mnb_12), list(mnb_21)
    rng.shuffle(list_a)
    rng.shuffle(list_b)
    out, ia, ib, turn_a = [], 0, 0, True
    while len(out) < n_total and (ia < len(list_a) or ib < len(list_b)):
        if turn_a and ia < len(list_a):
            out.append((list_a[ia], "q1_not_q2")); ia += 1
        elif not turn_a and ib < len(list_b):
            out.append((list_b[ib], "q2_not_q1")); ib += 1
        elif ia < len(list_a):
            out.append((list_a[ia], "q1_not_q2")); ia += 1
        elif ib < len(list_b):
            out.append((list_b[ib], "q2_not_q1")); ib += 1
        turn_a = not turn_a
    return out


def _proper_prefixes(x):
    return [x[:j] for j in range(1, len(x))]


def expand_with_prefixes(targets):
    """targets ∪ {every non-empty proper prefix each target needs}: the
    full set of literal label-strings actually put to the model for
    one history. The same set is asked under both s1 and s2 of a pair;
    only the ground truth (true_labels, below) differs by which node is
    being evaluated."""
    out = set(targets)
    for x in targets:
        out.update(_proper_prefixes(x))
    return out


def true_labels(candidates, dfa, q):
    """{candidate: bool}: ground truth (is_legal_from) for each
    candidate at node q. Purely offline."""
    return {x: is_legal_from(dfa, q, x) for x in candidates}


# --------------------------------------------------------------------------
# Prefix-closed derived acceptance
# --------------------------------------------------------------------------


def accepted_set(candidates, verdicts):
    """Prefix-closed derived acceptance from raw per-candidate verdicts.

    verdicts: {candidate: bool_or_None}, the parsed raw answer for
        each candidate (True/False), or None if the answer was missing
        or unparsable.

    A candidate counts as model-accepted only if its own raw verdict is
    True AND every one of its non-empty proper prefixes also has raw
    verdict True. Unlike L^W, the model's raw answers need not be
    prefix-closed; a candidate answered True whose own prefix was
    answered False is a closure violation, logged separately rather than
    silently repaired into either accepted or rejected.

    Returns (accepted, violations, missing), all sets of candidates:
      accepted:   prefix-closed derived acceptance holds.
      violations: candidate itself True, but a proper prefix False,
                  detected as soon as both of those two answers are
                  known, even if some other prefix in between is still
                  missing; a known contradiction doesn't need the rest
                  of the chain answered to already be a violation.
      missing:    no contradiction found yet, but the candidate's own
                  answer or a needed prefix's answer is still None.
    """
    accepted, violations, missing = set(), set(), set()
    for x in candidates:
        own = verdicts.get(x)
        prefix_verdicts = [verdicts.get(p) for p in _proper_prefixes(x)]
        if own and any(v is False for v in prefix_verdicts):
            violations.add(x)
        elif own is None or any(v is None for v in prefix_verdicts):
            missing.add(x)
        elif own:
            accepted.add(x)
    return accepted, violations, missing


# --------------------------------------------------------------------------
# Scoring (v1): compression (precision-shaped) + distinction recall only
# --------------------------------------------------------------------------


def score_compression(targets, verdicts_s1, verdicts_s2):
    """1 if every queried candidate (targets and their prefixes) has the
    same derived acceptance under s1 and s2, 0 if any differs: a
    disagreement on a prefix is still a real compression error for this
    pair, not something to drop just because it wasn't the target
    itself. Candidates missing on either side are excluded; if nothing
    is comparable, returns None."""
    candidates = expand_with_prefixes(targets)
    acc1, _, miss1 = accepted_set(candidates, verdicts_s1)
    acc2, _, miss2 = accepted_set(candidates, verdicts_s2)
    comparable = [x for x in candidates if x not in miss1 and x not in miss2]
    if not comparable:
        return None
    return 1 if all((t in acc1) == (t in acc2) for t in comparable) else 0


def score_distinction_recall(targets_with_direction, verdicts_s1, verdicts_s2):
    """Recall per direction: (correctly recognized) / (evaluable, sampled
    boundary words of that direction); never divided by the full,
    unsampled |MNB^W(q1,q2)|. verdicts_s1 answers the candidates asked
    under s1 (the history reaching q1), verdicts_s2 under s2 (reaching
    q2). A word is evaluable only if, for BOTH histories, it and all its
    needed prefixes have a well-formed answer.

    Returns {"q1_not_q2": {"recall": r_or_None, "n_evaluable": n},
             "q2_not_q1": {"recall": r_or_None, "n_evaluable": n}}.
    recall is None when n_evaluable == 0 for that direction: explicitly
    undefined, not 0 and not omitted.
    """
    targets = [t for t, _ in targets_with_direction]
    candidates = expand_with_prefixes(targets)
    acc1, _, miss1 = accepted_set(candidates, verdicts_s1)
    acc2, _, miss2 = accepted_set(candidates, verdicts_s2)

    def _recall(direction, correctly_recognized):
        evaluable = [t for t, d in targets_with_direction
                    if d == direction and t not in miss1 and t not in miss2]
        if not evaluable:
            return {"recall": None, "n_evaluable": 0}
        correct = sum(1 for t in evaluable if correctly_recognized(t))
        return {"recall": correct / len(evaluable),
               "n_evaluable": len(evaluable)}

    # q1_not_q2: true difference is accepted-via-s1, rejected-via-s2;
    # recognized correctly iff the model's derived acceptance matches.
    return {
        "q1_not_q2": _recall("q1_not_q2",
                             lambda t: t in acc1 and t not in acc2),
        "q2_not_q1": _recall("q2_not_q1",
                             lambda t: t in acc2 and t not in acc1),
    }


# --------------------------------------------------------------------------
# Diagnostics: computed from answers already collected, no extra calls,
# never feed back into the scores above
# --------------------------------------------------------------------------


def diagnostics_for_pair(targets, truth_s1, truth_s2, verdicts_s1, verdicts_s2):
    """Supplementary diagnostics for one probed pair. Does not affect
    score_compression / score_distinction_recall.

    truth_s1/truth_s2: {candidate: bool} from true_labels(), ground truth
        at the node each history reaches.
    verdicts_s1/verdicts_s2: {candidate: bool_or_None} raw model answers.

    Returns a dict:
      accuracy_by_length: {length: (n_correct, n_total)}, raw answers vs.
        ground truth, both histories combined.
      first_wrong_prefix_length: {(target, "s1"|"s2"): length_or_None}:
        shortest prefix length (walking the target's own chain from
        length 1) at which the raw answer first diverges from truth.
      first_divergent_acceptance_length: {target: length_or_None}:
        shortest length at which s1's and s2's derived, prefix-closed
        acceptance of that prefix first disagree; None if they never
        disagree within what was queried, or if data is missing before
        any disagreement is found.
      closure_violations: {"s1": set, "s2": set}, from accepted_set.
    """
    candidates = expand_with_prefixes(targets)
    acc1, viol1, miss1 = accepted_set(candidates, verdicts_s1)
    acc2, viol2, miss2 = accepted_set(candidates, verdicts_s2)

    acc_by_len = {}
    for truth, verdicts in ((truth_s1, verdicts_s1), (truth_s2, verdicts_s2)):
        for x in candidates:
            v = verdicts.get(x)
            if v is None:
                continue
            n_correct, n_total = acc_by_len.get(len(x), (0, 0))
            acc_by_len[len(x)] = (n_correct + (1 if v == truth[x] else 0),
                                  n_total + 1)

    first_wrong = {}
    for x in targets:
        for label, truth, verdicts in (("s1", truth_s1, verdicts_s1),
                                       ("s2", truth_s2, verdicts_s2)):
            found = None
            for j in range(1, len(x) + 1):
                p = x[:j]
                v = verdicts.get(p)
                if v is None:
                    continue
                if v != truth.get(p):
                    found = j
                    break
            first_wrong[(x, label)] = found

    first_divergent = {}
    for x in targets:
        found = None
        for j in range(1, len(x) + 1):
            p = x[:j]
            if p in miss1 or p in miss2:
                continue
            if (p in acc1) != (p in acc2):
                found = j
                break
        first_divergent[x] = found

    return {"accuracy_by_length": acc_by_len,
           "first_wrong_prefix_length": first_wrong,
           "first_divergent_acceptance_length": first_divergent,
           "closure_violations": {"s1": viol1, "s2": viol2}}


# --------------------------------------------------------------------------
# Dry-run oracle
# --------------------------------------------------------------------------


def dry_run_legality_reply(dfa, q, candidate):
    """Oracle single-elicit reply for one (q, candidate) pair: exactly
    what a perfectly correct model would answer. Used by --provider
    dry-run to keep the dry run offline and correct."""
    return json.dumps({"valid": is_legal_from(dfa, q, candidate)})


def make_dry_run_answer_fn(dfa, q):
    """answer_fn(messages, candidate) -> (raw_text, reasoning, usage)
    stand-in for --provider dry-run, bound to the node q that the
    relevant history is presumed to reach."""
    def answer_fn(messages, candidate):
        return dry_run_legality_reply(dfa, q, candidate), "", {}
    return answer_fn


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def parse_validity(raw, candidate):
    """Parse one single-elicit validity answer: a JSON object
    {"valid": true|false}; same status vocabulary as the other probe
    parsers. Takes the LAST parseable object (explore_agent's
    reason-then-answer convention, which the coherence system prompt
    also states), not the first: the first would return a stale verdict
    whenever the model reconsiders mid-reply."""
    obj = extract_last_json_object(raw)
    if obj is None:
        return {"status": "malformed_json"}
    if not isinstance(obj.get("valid"), bool):
        return {"status": "invalid_object"}
    return {"status": "ok", "valid": obj["valid"]}
