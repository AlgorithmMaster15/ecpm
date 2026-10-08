# Final run readiness

The passive `icl_expanded_v2` design is implemented. Its final audit uses
`icl_expanded_rows_v3`. Offline verification is not evidence that an endpoint
is admitted, that reasoning controls work there, or that results are correct.

## Passive ICL lock

- Five arms, both histories, separate deterministic/stochastic system prompts.
- Five graph seeds, eight states, K=10, all supported scenarios. Deterministic
  degradation is undefined and excluded.
- Four A-only balanced queries. Updating eligibility uses the submitted A
  route and its replay under B, including degradation and tied optima.
- Exact and tolerant beliefs, self/truth preservation, task/report consistency,
  preparation extraction review, raw requests/responses and filterable CSVs.
- One repeat initially; five repeats are an explicit later expansion. A seed
  label is not proof of provider seed support or independent graph sampling.

Use the commands in [the protocol guide](ICL_EXPANDED.md). A live command needs
an unused output path, a clean reviewed commit and a verified non-secret
deployment wrapper for the exact model, reasoning mode and history policy.
Generate the plan from those wrappers; do not run the illustrative matrix or
use test fixtures as readiness evidence. GPT-4o and DeepSeek have no expanded
request profiles yet. A generic provider option does not add such a profile.
The configured Sol and Gemma settings do not establish a shared greedy policy.

## Existing agentic workflow

`agentic_conditions.py` still has three arms. It is a different protocol:
model-first gets descriptions between episodes, while task-only does not get
matched neutral turns. The five-arm ICL change does not expand it automatically.

The current active runner explores M0 then M1 and asks probes after both phases.
It does not collect an A transition report before M1. Consequently, the existing
`--history-policy` option does not implement an agentic retained/separate
comparison. Explicit active history-policy and OFF/ON flags now fail before a
run starts, rather than being silently ignored. Do not label these runs as
either new history condition or verified OFF/ON.

`changed_action_usage.m0_route_uses` records whether the last M0 episode used
the changed action. It is null with no M0 episode or no changed pair. The exposure
CSV reuses this field and the existing phase counts. This is an exposure
descriptor, not proof that replanning was necessary. It is not interchangeable
with the ICL own-route replay criterion, especially for degradation.

The artifact and sidecar now share token normalization. Missing counts and
conflicting aliases give null totals with reported subtotals. Raw successful
usage reports remain in `provider_usage_calls` and `usage.json`. The accounting
scope is successful returned calls, not a complete billing ledger. Failed
provider attempts can consume tokens that this legacy path does not retain.
The recorded model-first system prompt now includes its final instruction.

Before a full agentic comparison, implement and review A/B report collection,
actual request logging (including trimming and correction/retry calls), explicit
reasoning/sampling controls and a versioned result export. Confirm the repeat
count and scenario/history matrix from the resulting executable plan.

`run_agentic_cost_check.py` is a small Azure cost pilot, not that full matrix.
Its defaults use one repeat and only silent break. Its completion check is
file-based, its rates are hard-coded, its cost output is cumulative, and it
does not safely sum null usage. Preserve existing pilot data; do not use it as
a full-study launcher or as verified model-independent billing evidence.

## Verification

Run the existing CI suites plus `python3 -B test_final_lock.py`. The latter
includes injected defects for false replanning, nullable reasoning metadata,
usage aliases/missing reports, the recorded system prompt and duplicated
exposure computation. Synthetic tests make no model calls.

Use a writable temporary directory outside the checkout when the system temp
directory is unavailable. Otherwise temporary synthetic artifacts can affect
the repository dirty-state provenance during tests. The legacy graph prompt
acceptance test additionally requires its six supplied prompt fixtures; an
absent fixture is not a pass.
