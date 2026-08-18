"""
Resource-Allocation / Packet-Routing MDP environment (ECPM, Phase 2) -- v2.

v1 responded to PI feedback (check-in 13/08/26): reframe the graph as a
resource-allocation / network problem -- ~8 nodes with transition
probabilities, where a link can break ("lost packet") -- and test whether
the model stops exploring the broken path.

v2 implements the Phase 2 technical-handoff alignment (Pavlos, 17/08/26):

  * Paired worlds ........ make_pair() emits frozen M0, intervened M1, and
                           an exact diff record (edge, old/new p, mode,
                           route position). M0 is never mutated.
  * Modes ................ deterministic=True routes through the SAME code
                           path (p_range=(1,1)); topology is identical to
                           the stochastic instance with the same seed.
  * Conditions ........... no_change | irrelevant | degradation |
                           silent_break | hard_removal, one argument.
  * Evidence ............. collect_balanced() guarantees >= k attempts per
                           listed (state, action) pair, with random starts;
                           paired_evidence() balances pre AND post and
                           emits verifiable per-pair counts.
  * Formats .............. same events rendered as F1 raw log, F2 ordered
                           triples, F2 shuffled triples (atomic facts),
                           F3 aggregated stats (explicit-model control),
                           F4 narrative -- with opaque per-node action
                           labels (a1, a2, ...) that never reveal the
                           destination. Labels are assigned on M0 and are
                           stable across the pair.
  * Oracle metadata ...... pre/post solvability, optimal route + exact
                           expected cost, tie/uniqueness info, and an
                           alternative proper route, per world.
  * Reproducibility ...... every random draw is seeded from explicit
                           string seeds (stable across processes); the
                           eligible break is randomized, not candidates[0].
  * Scoring seam ......... score_route() / route_from_actions() /
                           broken_link_usage() / regret() are the
                           environment-side primitives; the evaluator
                           computes headline metrics on top.

Grounding (unchanged from v1):

- Bertsekas & Tsitsiklis (1991), "An Analysis of Stochastic Shortest Path
  Problems", Mathematics of Operations Research 16(3). Finite SSP MDP:
  absorbing goal state, positive per-attempt cost, proper policies exist
  by construction.
- Boyan & Littman (1994), "Packet Routing in Dynamically Changing
  Networks: A Reinforcement Learning Approach", NeurIPS.

Key analytical convenience: with stay-on-failure semantics and cost 1 per
attempt, the expected number of attempts to cross link (i, j) is 1/p_ij,
so the optimal policy is a deterministic shortest path under edge weights
1/p_ij (Dijkstra). Exact optimal-cost baselines, no RL training.

Stdlib only. Python 3.8+.
"""

from __future__ import annotations

import heapq
import json
import random
import string
from collections import defaultdict
from dataclasses import dataclass, field

INF = float("inf")

CONDITIONS = ("no_change", "irrelevant", "degradation",
              "silent_break", "hard_removal")

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class RoutingMDP:
    """Directed graph; action at node u = pick an outgoing link (next hop).

    Transition: sending along (u, v) succeeds with prob p[(u, v)] -> packet
    moves to v; otherwise the packet is dropped and stays at u (retry).
    Cost: 1 per attempted transmission. Goal is absorbing.
    """

    def __init__(self, nodes, goal, p):
        self.nodes = list(nodes)
        self.goal = goal
        self.p = dict(p)      # (u, v) -> success probability
        self.broken = []      # v1 compat: edges broken via break_link
        self.changes = []     # v2: full intervention records (dicts)

    # -- construction ------------------------------------------------------

    @classmethod
    def generate(cls, n_nodes=8, seed=0, extra_edges=6,
                 p_range=(0.6, 0.95), obfuscate=False):
        """Random instance. A backbone chain guarantees every node can
        reach the goal; extra random edges create alternative routes.

        obfuscate=True swaps A..H for random 3-letter names (PlanBench-style
        ablation against prior knowledge).

        Deterministic mode is the SAME code path with p_range=(1.0, 1.0);
        because the rng consumes one uniform draw per added edge either
        way, the topology for a given seed is identical across modes.
        """
        rng = random.Random(seed)
        if obfuscate:
            names = []
            while len(names) < n_nodes:
                w = "".join(rng.choice(string.ascii_uppercase)
                            for _ in range(3))
                if w not in names:
                    names.append(w)
        else:
            assert n_nodes <= 26
            names = [chr(65 + i) for i in range(n_nodes)]

        order = names[:]
        rng.shuffle(order)
        goal = order[-1]

        p = {}

        def add(u, v):
            if u != v and u != goal and (u, v) not in p:
                p[(u, v)] = round(rng.uniform(*p_range), 2)
                return True
            return False

        for i in range(len(order) - 1):          # backbone toward goal
            add(order[i], order[i + 1])
        added, guard = 0, 0
        while added < extra_edges and guard < 1000:
            guard += 1
            if add(rng.choice(names), rng.choice(names)):
                added += 1
        # min out-degree 2 for non-goal nodes: a single link break must
        # leave an alternative, otherwise the V2 adaptation question
        # ("does it stop exploring that path?") is unanswerable.
        for u in names:
            if u == goal:
                continue
            guard = 0
            while sum(1 for (a, _) in p if a == u) < 2 and guard < 1000:
                guard += 1
                add(u, rng.choice(names))
        return cls(names, goal, p)

    def copy(self):
        """Independent copy (fresh intervention log). Used by make_pair so
        M0 is never mutated."""
        return RoutingMDP(self.nodes, self.goal, self.p)

    # -- dynamics ----------------------------------------------------------

    def out_edges(self, u):
        """Links listed at u (a silently-broken link is still listed)."""
        return sorted(v for (a, v) in self.p if a == u)

    def step(self, u, v, rng):
        """Attempt to send u -> v. Returns (next_node, success)."""
        if (u, v) not in self.p:
            raise ValueError(f"illegal action {u}->{v}")
        ok = rng.random() < self.p[(u, v)]
        return (v if ok else u), ok

    # -- interventions (V2) ------------------------------------------------

    def set_link_prob(self, u, v, new_p, mode="degrade"):
        """Generalized sparse perturbation with exact-diff bookkeeping.

        new_p = 0.0 .... silent break (link stays listed, always drops)
        new_p = None ... hard removal (link leaves the action set)
        else ........... degradation / arbitrary reweighting

        Records {edge, old_p, new_p, mode} in self.changes and returns it.
        """
        assert (u, v) in self.p, "cannot perturb a non-existent link"
        old = self.p[(u, v)]
        if new_p is None:
            del self.p[(u, v)]
        else:
            self.p[(u, v)] = new_p
        rec = {"edge": (u, v), "old_p": old, "new_p": new_p, "mode": mode}
        self.changes.append(rec)
        return rec

    def break_link(self, u, v, mode="silent"):
        """v1-compatible wrapper around set_link_prob.

        mode='silent': p -> 0 (link still listed; attempts always drop;
                       the model must *infer* the break from failures).
        mode='remove': link disappears from the action set (detection is
                       trivial; kept as an easier control)."""
        self.broken.append((u, v))
        if mode == "silent":
            return self.set_link_prob(u, v, 0.0, mode="silent")
        return self.set_link_prob(u, v, None, mode="remove")

    # -- exact solution (baselines) ----------------------------------------

    def optimal(self):
        """Expected attempts-to-goal per node + optimal next hop, via
        Dijkstra from the goal on reversed edges with weight 1/p."""
        dist = {u: INF for u in self.nodes}
        best = {}
        rev = defaultdict(list)
        for (u, v), pr in self.p.items():
            if pr > 0:
                rev[v].append((u, 1.0 / pr))
        dist[self.goal] = 0.0
        pq, seen = [(0.0, self.goal)], set()
        while pq:
            d, v = heapq.heappop(pq)
            if v in seen:
                continue
            seen.add(v)
            for (u, w) in rev[v]:
                nd = d + w
                if nd < dist[u] - 1e-12:
                    dist[u], best[u] = nd, v
                    heapq.heappush(pq, (nd, u))
        return dist, best

    def optimal_route(self, start):
        dist, best = self.optimal()
        if dist[start] == INF:
            return None, INF
        path, at = [start], start
        while at != self.goal and len(path) <= len(self.nodes):
            at = best[at]
            path.append(at)
        return path, dist[start]

    def plan_cost(self, path):
        """Exact expected cost of a proposed route; inf if invalid
        (missing/zero link or doesn't end at the goal). Used to score
        model-generated plans by *execution semantics*, not self-report."""
        if not path or path[-1] != self.goal:
            return INF
        total = 0.0
        for u, v in zip(path, path[1:]):
            pr = self.p.get((u, v), 0.0)
            if pr <= 0:
                return INF
            total += 1.0 / pr
        return total


# --------------------------------------------------------------------------
# Oracle metadata helpers
# --------------------------------------------------------------------------


def optimal_ties(mdp, tol=1e-9):
    """Nodes with more than one optimal next hop: {u: [v1, v2, ...]}.
    Basis for the tie/uniqueness field in the oracle metadata."""
    dist, _ = mdp.optimal()
    ties = {}
    for u in mdp.nodes:
        if u == mdp.goal or dist[u] == INF:
            continue
        opts = [v for v in mdp.out_edges(u)
                if mdp.p.get((u, v), 0.0) > 0
                and dist[v] < INF
                and abs(dist[u] - (1.0 / mdp.p[(u, v)] + dist[v])) < tol]
        if len(opts) > 1:
            ties[u] = opts
    return ties


def alternative_route(mdp, start):
    """Best proper route from `start` that avoids at least one edge of the
    current optimal route (zero each optimal edge in turn, re-solve, keep
    the cheapest). This is the 'alternative proper route' field of the
    oracle metadata. Returns (route, cost) or (None, inf)."""
    route, _ = mdp.optimal_route(start)
    if route is None:
        return None, INF
    best_alt, best_cost = None, INF
    for u, v in zip(route, route[1:]):
        saved = mdp.p[(u, v)]
        mdp.p[(u, v)] = 0.0
        r2, c2 = mdp.optimal_route(start)
        mdp.p[(u, v)] = saved
        if r2 is not None and c2 < best_cost:
            best_alt, best_cost = r2, c2
    return best_alt, best_cost


def breakable_route_links(mdp, start):
    """Links on the current optimal route whose silent break keeps the goal
    reachable from `start` -- i.e. perturbations that force *replanning*
    rather than impossibility. The eligible-break set for V2 episodes."""
    route, _ = mdp.optimal_route(start)
    if route is None:
        return []
    keep = []
    for u, v in zip(route, route[1:]):
        saved = mdp.p[(u, v)]
        mdp.p[(u, v)] = 0.0
        dist, _ = mdp.optimal()
        if dist[start] < INF:
            keep.append((u, v))
        mdp.p[(u, v)] = saved
    return keep


# --------------------------------------------------------------------------
# Opaque action labels (stable across the pair; never reveal destinations)
# --------------------------------------------------------------------------


def assign_labels(mdp, label_rng):
    """Per-node opaque labels: at node u the listed out-links get a1..a_d
    in an rng-shuffled order, so the label string carries no information
    about the destination. Returns {(u, v): 'aK'}.

    Assign on M0 and reuse for M1: interventions never add edges, so every
    M1-listed pair is covered, and labels do not shift after a hard
    removal (which would otherwise leak the change through renumbering)."""
    labels = {}
    for u in mdp.nodes:
        outs = mdp.out_edges(u)
        if not outs:
            continue
        perm = label_rng.sample(outs, len(outs))
        for i, v in enumerate(perm):
            labels[(u, v)] = f"a{i + 1}"
    return labels


def invert_labels(labels):
    """{(u, 'aK') -> v} for translating model action-plans back to nodes."""
    return {(u, lab): v for (u, v), lab in labels.items()}


def legal_actions(mdp, labels):
    """{u: ['a1', 'a2', ...]} for the listed out-links of each node --
    the model-visible action menu (differs between M0/M1 only under
    hard_removal, where the removed label disappears)."""
    menu = {}
    for u in mdp.nodes:
        outs = mdp.out_edges(u)
        if outs:
            menu[u] = sorted((labels[(u, v)] for v in outs),
                             key=lambda s: int(s[1:]))
    return menu


def route_from_actions(start, actions, labels):
    """Translate an action-label plan ['a2', 'a1', ...] from `start` into a
    node path using the pair's label mapping; None if any label is not
    defined at the node reached so far. Score the result with score_route
    (a removed edge translates but then costs inf, i.e. invalid)."""
    inv = invert_labels(labels)
    path, at = [start], start
    for lab in actions:
        v = inv.get((at, lab))
        if v is None:
            return None
        path.append(v)
        at = v
    return path


# --------------------------------------------------------------------------
# Paired pre/post-change instances (M0, M1, exact diff, oracle metadata)
# --------------------------------------------------------------------------


@dataclass
class PairedInstance:
    condition: str
    deterministic: bool
    m0: RoutingMDP          # frozen pre-change world
    m1: RoutingMDP          # post-change world (equals m0 under no_change)
    start: str
    labels: dict            # (u, v) -> 'aK', assigned on M0, stable
    change: dict            # exact diff (edge None under no_change)
    oracle: dict            # pre/post solvability, routes, costs, ties, alt
    seeds: dict
    params: dict


def _world_oracle(mdp, start, labels):
    route, cost = mdp.optimal_route(start)
    solvable = route is not None
    ties = optimal_ties(mdp)
    tie_on = [u for u in (route[:-1] if solvable else []) if u in ties]
    alt, alt_cost = alternative_route(mdp, start) if solvable else (None, INF)
    acts = ([labels[(a, b)] for a, b in zip(route, route[1:])]
            if solvable else None)
    alt_acts = ([labels[(a, b)] for a, b in zip(alt, alt[1:])]
                if alt else None)
    return {
        "solvable": solvable,
        "optimal_route": route,
        "optimal_actions": acts,
        "optimal_cost": round(cost, 4) if solvable else None,
        "route_unique": bool(solvable and not tie_on),
        "tie_nodes_on_route": tie_on,
        "alternative_route": alt,
        "alternative_actions": alt_acts,
        "alternative_cost": round(alt_cost, 4) if alt else None,
    }


def build_oracle(m0, m1, start, labels):
    pre = _world_oracle(m0, start, labels)
    post = _world_oracle(m1, start, labels)
    delta = (round(post["optimal_cost"] - pre["optimal_cost"], 4)
             if pre["solvable"] and post["solvable"] else None)
    return {"start": start, "goal": m0.goal, "pre": pre, "post": post,
            "route_changed": pre["optimal_route"] != post["optimal_route"],
            "cost_delta": delta}


def make_pair(seed, condition, *, deterministic=False, n_nodes=8,
              extra_edges=6, p_range=(0.6, 0.95), obfuscate=False,
              degradation_factor=0.5, change_seed=None):
    """Generate a paired instance (M0, M1, exact diff, oracle metadata).

    condition:
      no_change ...... M1 = M0 (false-positive control)
      irrelevant ..... degrade one link OFF the optimal route (guaranteed
                       not to alter the optimal route: off-route costs can
                       only rise, so the optimum is preserved)
      degradation .... degrade one link ON the optimal route
                       (new_p = max(0.05, old_p * degradation_factor))
      silent_break ... p -> 0 on an eligible optimal-route link (still
                       listed; goal stays reachable, forcing replanning)
      hard_removal ... same eligible set, link removed from the action set

    deterministic=True sets p_range=(1.0, 1.0) through the same code path
    (identical topology per seed). The eligible break is drawn at RANDOM
    from breakable_route_links via a string-seeded rng -- pass a different
    change_seed to redraw on the same graph. Raises ValueError if the seed
    admits no eligible break (pick another seed).
    """
    assert condition in CONDITIONS, f"unknown condition {condition!r}"
    if deterministic:
        p_range = (1.0, 1.0)

    m0 = RoutingMDP.generate(n_nodes=n_nodes, seed=seed,
                             extra_edges=extra_edges, p_range=p_range,
                             obfuscate=obfuscate)
    dist0, _ = m0.optimal()
    starts = [n for n in m0.nodes if n != m0.goal and dist0[n] < INF]
    start = max(starts, key=lambda n: (dist0[n], n))   # farthest, det. tie
    route0, _ = m0.optimal_route(start)
    route0_edges = list(zip(route0, route0[1:]))

    cs = f"{seed}|{condition}|{'det' if deterministic else 'sto'}|{change_seed}"
    crng = random.Random(cs)
    ls = f"{seed}|labels"
    labels = assign_labels(m0, random.Random(ls))

    m1 = m0.copy()
    if condition == "irrelevant":
        off = sorted(e for e in m0.p if e not in set(route0_edges))
        if not off:
            raise ValueError(f"seed {seed}: no off-route link to perturb")
        u, v = crng.choice(off)
        old = m0.p[(u, v)]
        new = max(0.05, round(old * degradation_factor, 2))
        if new >= old:
            new = round(old / 2.0, 3)
        m1.set_link_prob(u, v, new, mode="degrade")
    elif condition == "degradation":
        u, v = crng.choice(route0_edges)
        old = m0.p[(u, v)]
        new = max(0.05, round(old * degradation_factor, 2))
        if new >= old:
            new = round(old / 2.0, 3)
        m1.set_link_prob(u, v, new, mode="degrade")
    elif condition in ("silent_break", "hard_removal"):
        cands = breakable_route_links(m1, start)
        if not cands:
            raise ValueError(
                f"seed {seed}: no eligible break keeps the goal reachable; "
                f"use a different seed")
        u, v = crng.choice(cands)
        m1.break_link(u, v,
                      mode="silent" if condition == "silent_break"
                      else "remove")
    # condition == "no_change": leave m1 untouched

    change = {"edge": None, "action": None, "old_p": None, "new_p": None,
              "mode": "none", "on_optimal_route": None,
              "route_position": None}
    if m1.changes:
        rec = m1.changes[-1]
        e = rec["edge"]
        on_route = e in route0_edges
        change = {"edge": e, "action": labels[e],
                  "old_p": rec["old_p"], "new_p": rec["new_p"],
                  "mode": rec["mode"], "on_optimal_route": on_route,
                  "route_position": route0_edges.index(e) if on_route
                  else None}

    oracle = build_oracle(m0, m1, start, labels)
    seeds = {"graph_seed": seed, "change_seed": cs, "label_seed": ls}
    params = {"n_nodes": n_nodes, "extra_edges": extra_edges,
              "p_range": list(p_range), "obfuscate": obfuscate,
              "degradation_factor": degradation_factor}
    return PairedInstance(condition, deterministic, m0, m1, start, labels,
                          change, oracle, seeds, params)


# --------------------------------------------------------------------------
# Rollouts and data collection
# --------------------------------------------------------------------------


@dataclass
class Attempt:
    t: int
    node: str
    chosen: str
    success: bool
    next_node: str


def rollout(mdp, start, policy, rng, horizon=40):
    """policy: callable(node, rng) -> next hop. Returns (attempts, delivered)."""
    at, t, attempts = start, 0, []
    while at != mdp.goal and t < horizon:
        v = policy(at, rng)
        nxt, ok = mdp.step(at, v, rng)
        t += 1
        attempts.append(Attempt(t, at, v, ok, nxt))
        at = nxt
    return attempts, at == mdp.goal


def optimal_policy(mdp):
    _, best = mdp.optimal()

    def pol(u, rng):
        v = best.get(u)
        outs = mdp.out_edges(u)
        return v if v in outs else outs[0]
    return pol


def explore_policy(mdp, eps=0.5):
    """v1 eps-greedy baseline collector (kept for comparison runs; the
    primary V2 collector is collect_balanced)."""
    _, best = mdp.optimal()

    def pol(u, rng):
        outs = mdp.out_edges(u)
        if u in best and best[u] in outs and rng.random() > eps:
            return best[u]
        return rng.choice(outs)
    return pol


def fixed_policy(best_map, fallback_mdp):
    """Freeze a next-hop table (e.g. the PRE-change optimum) -- the 'stale
    planner' baseline for V2 adaptation metrics. Falls back to the first
    listed link if the frozen hop is no longer legal (hard removal)."""
    def pol(u, rng):
        v = best_map.get(u)
        outs = fallback_mdp.out_edges(u)
        return v if v in outs else outs[0]
    return pol


def collect_balanced(mdp, k, rng, horizon=60, max_episodes=5000):
    """Coverage-balanced collection: episodic rollouts with RANDOM starts
    and a least-tried action rule, run until every listed (state, action)
    pair has >= k attempts. The data budget is therefore expressed per
    state-action pair, not per episode.

    Returns (episodes, counts) where counts[(u, v)] is the number of
    attempts, so the balance guarantee is externally verifiable. A
    silently-broken link is still listed, hence it too receives >= k
    attempts post-change (all drops -- exactly the detection evidence).
    Raises RuntimeError if max_episodes is exhausted first.
    """
    counts = {e: 0 for e in mdp.p}
    starts = [n for n in mdp.nodes if n != mdp.goal]

    def pol(u, r):
        outs = mdp.out_edges(u)
        m = min(counts[(u, w)] for w in outs)
        choice = r.choice([w for w in outs if counts[(u, w)] == m])
        counts[(u, choice)] += 1          # the attempt will happen
        return choice

    episodes = []
    while counts and min(counts.values()) < k:
        if len(episodes) >= max_episodes:
            raise RuntimeError("coverage not reached; raise max_episodes "
                               "or horizon")
        s = rng.choice(starts)
        att, _ = rollout(mdp, s, pol, rng, horizon)
        episodes.append(att)
    return episodes, counts


def attempt_stats(episodes):
    """Per-pair (attempts, deliveries) across a list of episodes."""
    tries, okc = defaultdict(int), defaultdict(int)
    for ep in episodes:
        for a in ep:
            tries[(a.node, a.chosen)] += 1
            okc[(a.node, a.chosen)] += a.success
    return tries, okc


# --------------------------------------------------------------------------
# Exposure-format emitters (same events, different formats)
# --------------------------------------------------------------------------


def _act(a, labels):
    return labels[(a.node, a.chosen)] if labels else f"->{a.chosen}"


def emit_f1_log(attempts, labels=None):
    """F1 -- raw step log (ordered trajectory)."""
    lines = []
    for a in attempts:
        out = (f"OK, arrive {a.next_node}" if a.success
               else f"DROP, stay {a.node}")
        lines.append(f"t={a.t} | at {a.node} | send {_act(a, labels)} | {out}")
    return "\n".join(lines)


def emit_f2_triples(attempts, labels=None):
    """F2 -- bare (state, action, next-state) triples."""
    return "\n".join(f"({a.node}, {_act(a, labels)}, {a.next_node})"
                     for a in attempts)


def emit_f3_stats(list_of_attempt_lists, labels=None):
    """F3 -- aggregated per-link statistics ('facts' end of the
    facts-vs-trajectories axis; the explicit-model control)."""
    tries, okc = attempt_stats(list_of_attempt_lists)
    lines = []
    for (u, v) in sorted(tries):
        n, s = tries[(u, v)], okc[(u, v)]
        if labels:
            lab = labels[(u, v)]
            key, order = f"{u} {lab}", (u, int(lab[1:]))
        else:
            key, order = f"{u}->{v}", (u, v)
        lines.append((order, f"{key}: {n} attempts, {s} delivered "
                             f"({s / n:.2f})"))
    return "\n".join(line for _, line in sorted(lines))


def emit_f4_narrative(attempts, labels=None):
    """F4 -- natural-language episodic narrative."""
    lines = []
    for a in attempts:
        act = (f"via action {labels[(a.node, a.chosen)]}" if labels
               else f"toward {a.chosen}")
        if a.success:
            lines.append(f"At {a.node}, the packet was sent {act} "
                         f"and arrived at {a.next_node}.")
        else:
            lines.append(f"At {a.node}, the packet was sent {act}, "
                         f"but the link dropped it and it stayed at "
                         f"{a.node}.")
    return "\n".join(lines)


def render_evidence(episodes, labels=None, shuffle_seed="0"):
    """One event set, five renderings:
      F1_log ........ ordered raw logs, episode-separated
      F2_ordered .... triples in trajectory order, episode-separated
      F2_shuffled ... the SAME triples pooled and shuffled (atomic facts)
      F3_stats ...... aggregated per-link statistics (explicit model)
      F4_narrative .. ordered natural-language narrative
    """
    eps = [ep for ep in episodes if ep]
    f1 = "\n".join(f"--- episode {i + 1} ---\n{emit_f1_log(ep, labels)}"
                   for i, ep in enumerate(eps))
    f2o = "\n\n".join(emit_f2_triples(ep, labels) for ep in eps)
    pooled = [ln for ep in eps for ln in
              emit_f2_triples(ep, labels).split("\n")]
    random.Random(shuffle_seed).shuffle(pooled)
    f4 = "\n".join(f"--- episode {i + 1} ---\n{emit_f4_narrative(ep, labels)}"
                   for i, ep in enumerate(eps))
    return {"F1_log": f1, "F2_ordered": f2o, "F2_shuffled": "\n".join(pooled),
            "F3_stats": emit_f3_stats(eps, labels), "F4_narrative": f4}


def paired_evidence(inst, k=5, evidence_seed=0, horizon=60,
                    max_episodes=5000):
    """Balanced pre AND post evidence for a PairedInstance: >= k attempts
    per listed (state, action) pair in each world, all five renderings,
    plus verifiable per-pair counts."""
    r0 = random.Random(f"{evidence_seed}|pre")
    r1 = random.Random(f"{evidence_seed}|post")
    ep0, c0 = collect_balanced(inst.m0, k, r0, horizon, max_episodes)
    ep1, c1 = collect_balanced(inst.m1, k, r1, horizon, max_episodes)
    t0, s0 = attempt_stats(ep0)
    t1, s1 = attempt_stats(ep1)

    def counts_json(t, s):
        return {f"{u}->{v}": {"attempts": t[(u, v)],
                              "delivered": s[(u, v)]}
                for (u, v) in sorted(t)}

    return {
        "k_per_pair": k,
        "evidence_seed": evidence_seed,
        "episodes_pre": len(ep0),
        "episodes_post": len(ep1),
        "min_attempts_pre": min(c0.values()) if c0 else 0,
        "min_attempts_post": min(c1.values()) if c1 else 0,
        "pre": render_evidence(ep0, inst.labels, f"{evidence_seed}|shuf-pre"),
        "post": render_evidence(ep1, inst.labels,
                                f"{evidence_seed}|shuf-post"),
        "counts_pre": counts_json(t0, s0),
        "counts_post": counts_json(t1, s1),
    }


# --------------------------------------------------------------------------
# Scoring seam (environment-side primitives; evaluator builds on these)
# --------------------------------------------------------------------------


def score_route(mdp, path, start):
    """Score a proposed node route by execution semantics, never
    self-report: validity, exact expected cost, and regret vs the exact
    optimum. `path` may be None (parse failure) -> invalid."""
    dist, _ = mdp.optimal()
    opt = dist.get(start, INF)
    cost = mdp.plan_cost(path) if path else INF
    valid = bool(path) and path[0] == start and cost < INF
    return {
        "valid": valid,
        "expected_cost": round(cost, 4) if valid else None,
        "optimal_cost": round(opt, 4) if opt < INF else None,
        "regret": round(cost - opt, 4) if valid and opt < INF else None,
    }


def broken_link_usage(attempts, u, v):
    """Among post-change visits to u, share of decisions still choosing v.
    ~0 for an adapted agent; ~1 for a stale one. None if u never visited.
    (PI criterion: 'see if the model stops exploring that path.')"""
    visits = [a for a in attempts if a.node == u]
    if not visits:
        return None
    return sum(1 for a in visits if a.chosen == v) / len(visits)


def regret(mdp, path, start):
    """Expected cost of a proposed route minus the optimal expected cost."""
    dist, _ = mdp.optimal()
    return mdp.plan_cost(path) - dist[start]


# --------------------------------------------------------------------------
# JSON serialization (the interface Pavlos's prompts/evaluator consume)
# --------------------------------------------------------------------------

SCHEMA_VERSION = "2.0"


def _edges_json(mdp, labels):
    return [{"from": u, "to": v, "p": mdp.p[(u, v)],
             "action": labels.get((u, v))}
            for (u, v) in sorted(mdp.p)]


def pair_to_json(inst, evidence=None):
    """Full instance record. `model_visible` lists the only fields that may
    reach the prompt; everything else is evaluator-only ground truth."""
    ch = inst.change
    return {
        "schema_version": SCHEMA_VERSION,
        "condition": inst.condition,
        "deterministic": inst.deterministic,
        "seeds": inst.seeds,
        "params": inst.params,
        "nodes": inst.m0.nodes,
        "goal": inst.m0.goal,
        "start": inst.start,
        "legal_actions_pre": legal_actions(inst.m0, inst.labels),
        "legal_actions_post": legal_actions(inst.m1, inst.labels),
        "world_pre": {"edges": _edges_json(inst.m0, inst.labels)},
        "world_post": {"edges": _edges_json(inst.m1, inst.labels)},
        "change": {
            "edge": (None if ch["edge"] is None
                     else {"from": ch["edge"][0], "to": ch["edge"][1]}),
            "action": ch["action"],
            "old_p": ch["old_p"],
            "new_p": ch["new_p"],
            "mode": ch["mode"],
            "on_optimal_route": ch["on_optimal_route"],
            "route_position": ch["route_position"],
        },
        "oracle": inst.oracle,
        "evidence": evidence,
        "model_visible": ["nodes", "start", "goal", "legal_actions_pre",
                          "legal_actions_post", "evidence"],
        "evaluator_only": ["world_pre", "world_post", "change", "oracle",
                           "seeds", "params", "counts_*"],
    }


# --------------------------------------------------------------------------
# Demo: the two end-to-end paired examples for the meeting
# --------------------------------------------------------------------------


def _summarize(inst, ev):
    o = inst.oracle
    print(f"[{('deterministic' if inst.deterministic else 'stochastic')}"
          f" | {inst.condition}]  seed {inst.seeds['graph_seed']}")
    print(f"  goal {o['goal']}  start {o['start']}")
    print(f"  change: {inst.change['edge']} "
          f"(action {inst.change['action']}) "
          f"p {inst.change['old_p']} -> {inst.change['new_p']} "
          f"[{inst.change['mode']}], route position "
          f"{inst.change['route_position']}")
    print(f"  pre : route {o['pre']['optimal_route']}  "
          f"cost {o['pre']['optimal_cost']}  "
          f"unique {o['pre']['route_unique']}")
    print(f"  post: route {o['post']['optimal_route']}  "
          f"cost {o['post']['optimal_cost']}  "
          f"solvable {o['post']['solvable']}")
    print(f"  alternative (pre): {o['pre']['alternative_route']}  "
          f"cost {o['pre']['alternative_cost']}")
    print(f"  balance: min attempts per pair pre={ev['min_attempts_pre']} "
          f"post={ev['min_attempts_post']}  "
          f"episodes {ev['episodes_pre']}/{ev['episodes_post']}")
    u, v = inst.change["edge"]
    rng = random.Random("demo|metrics")
    pre_best = inst.m0.optimal()[1]
    stale = fixed_policy(pre_best, inst.m1)
    adapted = optimal_policy(inst.m1)
    stale_att = [a for _ in range(20)
                 for a in rollout(inst.m1, inst.start, stale, rng, 40)[0]]
    adapt_att = [a for _ in range(20)
                 for a in rollout(inst.m1, inst.start, adapted, rng, 40)[0]]
    su = broken_link_usage(stale_att, u, v)
    au = broken_link_usage(adapt_att, u, v)
    print(f"  broken-link usage at {u}: stale "
          f"{'n/a' if su is None else f'{su:.2f}'}, adapted "
          f"{'n/a' if au is None else f'{au:.2f}'}")
    sc = score_route(inst.m1, o["post"]["optimal_route"], inst.start)
    print(f"  score_route(post-optimal) -> {sc}")
    print()


if __name__ == "__main__":
    import os
    outdir = os.environ.get("OUT", ".")
    for tag, det in (("deterministic", True), ("stochastic", False)):
        inst = make_pair(seed=7, condition="silent_break", deterministic=det)
        ev = paired_evidence(inst, k=5, evidence_seed=11)
        _summarize(inst, ev)
        path = os.path.join(outdir, f"example_{tag}_silent_break.json")
        with open(path, "w") as f:
            json.dump(pair_to_json(inst, ev), f, indent=2)
        print(f"  wrote {path}\n")
    inst = make_pair(seed=7, condition="silent_break")
    ev = paired_evidence(inst, k=3, evidence_seed=2)
    print("--- F1 sample (first 6 lines, opaque actions) ---")
    print("\n".join(ev["pre"]["F1_log"].split("\n")[:7]))
    print("\n--- F3 aggregated statistics (post-change; note the broken "
          "pair at 0.00) ---")
    print(ev["post"]["F3_stats"])
