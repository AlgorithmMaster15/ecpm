# Routing-MDP Environment Interface (v2.1) — for prompts, parser, and evaluator

Module: `resource_mdp.py` (stdlib only, Python 3.8+). Schema version **2.1**.
Response parsing/scoring: `ecpm_parser.py` + `parser_fixtures.json` +
`test_ecpm_parser.py` (Section 7 is now FROZEN, not a proposal).
The two example files (`example_deterministic_silent_break.json`,
`example_stochastic_silent_break.json`) are instances of exactly this
schema: seed 7, condition `silent_break`, `k=5`, `evidence_seed=0`.

v2.1 resolves the six pre-freeze review items (2026-08-19): matched
targets, strict irrelevant control, binary deterministic mode, pure
relabeling obfuscation, distinct score statuses, capped evidence with a
prompt-safe projection, and a frozen response schema.

## 1. Handoff table → implementation

| Handoff request | Where it lives |
| --- | --- |
| Paired worlds, exact diff, old/new p | `make_pair(seed, condition, ...)` → `PairedInstance(m0, m1, change, ...)`; M0 is never mutated |
| Deterministic + stochastic, same interface | `deterministic=True` sets `p_range=(1,1)` through the same code path; **topology is identical per seed across modes** |
| Five conditions | `condition ∈ {no_change, irrelevant, degradation, silent_break, hard_removal}` (`degradation` is stochastic-only, see §4) |
| **Matched break target (v2.1)** | `silent_break` and `hard_removal` draw from ONE shared target stream → same link per (seed, mode) |
| Balanced per-(s,a) pre/post evidence | `paired_evidence(inst, k, evidence_seed)` — ≥ k attempts per listed pair in each world; capped episodes + deterministic resampling (§6) |
| **Prompt-safe projection (v2.1)** | `prompt_view(record, rendering, periods, budget_per_pair)` — the ONLY object a prompt may serialize (§6) |
| Oracle metadata | `PairedInstance.oracle` — pre/post solvability, optimal route/cost, tie info, alternative proper route |
| Reproducibility | all rngs seeded with explicit strings; records are self-describing (a record can be rebuilt from its `seeds`/`params`) |
| Scoring seam | `score_route` (now with status, §7), `route_from_actions`, `broken_link_usage`, `regret`; response parsing in `ecpm_parser.py` |

## 2. Core API (field names to code against)

```python
inst = make_pair(seed, condition, deterministic=False, n_nodes=8,
                 extra_edges=6, p_range=(0.6, 0.95), obfuscate=False,
                 degradation_factor=0.5, change_seed=None)
# ValueError: deterministic degradation (undefined in v2.1), or no
# eligible target for the requested condition on this seed.

ev = paired_evidence(inst, k=5, evidence_seed=0, horizon=60,
                     max_episodes=300, max_resamples=4)
record = pair_to_json(inst, ev)          # the full JSON instance (§3)
view = prompt_view(record, rendering="F2_shuffled",
                   periods=("pre", "post"), budget_per_pair=5)
```

`RoutingMDP` internals the evaluator may touch: unchanged from v2.0
(`.nodes .goal .p .broken .changes .out_edges .optimal .optimal_route
.plan_cost .set_link_prob .break_link .copy`). New helpers:
`forward_distances(mdp, start)` and `edges_off_all_optimal_routes(mdp,
start)` (the v2.1 irrelevant eligibility set). Rollout side unchanged.
Obfuscation now draws names from a separate stream, so `obfuscate=True`
is pure relabeling (identical indexed weighted graph).

## 3. JSON instance — top level

| Field | Content |
| --- | --- |
| `schema_version` | "2.1" |
| `condition`, `deterministic` | condition name; mode flag |
| `seeds`, `params` | full provenance; sufficient to rebuild the instance |
| `nodes`, `goal`, `start` | start = farthest solvable node |
| `legal_actions_pre/post` | model-visible action menus (differ only under `hard_removal`) |
| `world_pre/post` | ground-truth graphs (evaluator-only) |
| `change` | exact diff (§4) |
| `oracle` | metadata bundle (§5) |
| `evidence` | balanced observations + counts (§6) — **evaluator-only in v2.1** |
| `model_visible` | `nodes, start, goal, legal_actions_*` — static prompt fields |
| `prompt_projection` | builder contract: prompts are built with `prompt_view` only |
| `evaluator_only` | `world_*, change, oracle, seeds, params, evidence, counts_*` |

## 4. `change` — the exact diff (localization ground truth)

Format unchanged from v2.0. Condition semantics (v2.1):

| condition | target | effect | guarantees |
| --- | --- | --- | --- |
| `no_change` | — | M1 = M0 | false-positive control |
| `irrelevant` | random edge off **every** optimal route | stochastic: p × `degradation_factor`; deterministic: p → 0 (binary) | optimal route AND `route_unique` provably unchanged; deterministic post world stays {0, 1} |
| `degradation` | random ON-route edge | p × `degradation_factor` (min 0.05) | **stochastic-only**; deterministic raises ValueError |
| `silent_break` | random eligible ON-route edge (shared stream) | p → 0, still listed | goal stays reachable → forces replanning |
| `hard_removal` | **same target as silent_break** per (seed, mode) | edge leaves the action set | detection trivial by menu diff — the easier control |

Eligibility for breaks = `breakable_route_links` (reachability preserved);
for `irrelevant` = `edges_off_all_optimal_routes`. `make_pair` raises
ValueError on seeds with an empty eligible set.

## 5. `oracle` — evaluator ground truth

Unchanged from v2.0. Reminder: any `valid_finite` route with regret == 0
is scored optimal; exact equality with `optimal_route` is NOT required,
and `route_unique` stays diagnostic.

## 6. `evidence` and the prompt budget — what the model sees

`pre`/`post` carry the SAME event set in five renderings (`F1_log`,
`F2_ordered`, `F2_shuffled`, `F3_stats`, `F4_narrative`), plus
`counts_pre/post`, `min_attempts_pre/post`, `episodes_pre/post`,
`k_per_pair`, `evidence_seed`. New v2.1 fields: `evidence_seed_effective`,
`resamples`, `horizon`, `max_episodes`.

**Budget rule (v2.1).** Collection is capped at `max_episodes` (default
300) per world; on failure it deterministically resamples with a derived
seed up to `max_resamples` times, then raises. `k` remains a minimum-
coverage gate, NOT the prompt budget.

**Prompt contract (v2.1).** The raw `evidence` object is evaluator-only:
it contains `counts_*` keyed by true edges and all five renderings.
Prompts are built exclusively with

```python
prompt_view(record, rendering, periods, budget_per_pair=5, budget_seed=0)
```

which returns ONE rendering of the selected period(s), the matching
action menus, `nodes/start/goal`, and realized sizes (`events`,
`episodes`, `chars` per period) — never counts, worlds, change, oracle,
or seeds. Events are subsampled to an EXACT per-pair budget (uniform per
pair, outcome-blind, deterministic given `budget_seed`); the same
subsampled event set underlies every rendering choice, so the
format comparison stays matched, and the silently broken pair still
contributes exactly `budget_per_pair` all-drop observations post-change.
Original `t` values are preserved, so elisions in `F1_log`/`F4` appear as
t-jumps inside an episode. Log `realized` for every prompt.

## 7. Response schema, parser, and scoring — FROZEN

Implementation: `ecpm_parser.py` (stdlib; consumes raw model text + the
JSON record). Acceptance cases: `parser_fixtures.json` (format level) and
`test_ecpm_parser.py` (instance level). One JSON object per probe; the
FIRST balanced `{...}` object in the reply is parsed; surrounding prose
and code fences are ignored; unknown extra fields are ignored.

1. **Detection** — `{"changed": true|false}`. Truth:
   `condition != "no_change"` (`irrelevant` counts as changed).
2. **Localization** — `{"node": "C", "action": "a2"}`. Truth:
   `change.edge.from` + `change.action`.
3. **Preservation** — `{"pairs": [{"node", "action", "changed"}, ...]}`
   (optional numeric `"p"` per pair recorded). Truth: a pair changed iff
   it is the intervention target; scored as accuracy over the queried
   pairs.
4. **Adaptation** — `{"route": [{"node", "action"}, ...]}`, the canonical
   representation: explicit state-action steps, ≤ 32 steps, starting at
   `start`; step i+1's node must equal the destination implied by step i
   (destinations resolved on the PRE world, so a removed edge translates
   and then scores `illegal_action` on the POST world).

Statuses:

| Layer | Statuses |
| --- | --- |
| parse | `ok`, `malformed_json`, `invalid_object`, `too_long` |
| route walk | `unknown_reference`, `discontinuous_route`, `incomplete_route` |
| execution (POST world) | `valid_finite` (cost + regret), `silent_broken_edge` (legal route over a listed p = 0 link), `illegal_action` (hop not in the action set) |

`score_route` returns the same execution statuses plus `invalid_route`
for empty/mis-anchored node paths; `valid` (bool) is kept for v2.0
compatibility. A `valid_finite` route with regret == 0 → `is_optimal`.

## 8. Invariants the tests enforce

`test_resource_mdp.py` (v2.0 battery, all kept) plus v2.1: silent break
and hard removal share the target per (seed, mode); the irrelevant target
is off every optimal route, preserves `route_unique`, and keeps the
deterministic post world binary; deterministic degradation raises;
obfuscation preserves the indexed weighted graph; score statuses are
distinct; `prompt_view` leaks nothing evaluator-only, enforces the exact
per-pair budget, and reproduces the stored renderings at unbounded
budget; the shipped seed-7 examples regenerate exactly.
`test_ecpm_parser.py` covers the frozen Section 7 contract end to end.

Run: `python3 test_resource_mdp.py && python3 test_ecpm_parser.py`
Audit: `python3 ecpm_pre_freeze_audit.py` (Pavlos, read-only) and
`python3 ecpm_reply_verification.py` (regression numbers for the v2.1
fixes; against v2.0 it reproduced the review findings).
