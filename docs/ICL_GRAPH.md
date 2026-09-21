# ICL graph availability

`icl_graph_availability_v1` is an additive, passive follow-up on the known
deterministic silent-break worlds at seeds 8, 13 and 25. It does not change
older protocols or the six seed-8 prompts.

| Condition | Period A | Period B |
| --- | --- | --- |
| `graph_ab` | Current graph and logs | Current graph and logs |
| `graph_a` | Current graph and logs | Logs; A remains in history |
| `logs_only` | Logs | Logs; A remains in history |

Only the current graph blocks differ. All conditions use the same mechanics,
shuffled observations, five questions and action-only route contract.
A final answer is replayed verbatim into B, even if malformed; private
reasoning is not replayed. The original first-object parser and scorer apply.

Use [the offline preview script](../experiments/preview_icl_graph.py) to generate
all six full prompts, sequential graph/log reference answers, hashes and
explicitly assumed context/cost bounds outside Git:

```sh
python3 -B experiments/preview_icl_graph.py --out /tmp/icl_graph_review
python3 -B test_icl_graph.py
python3 -B run_pilot.py --protocol icl_graph_availability_v1 \
  --scenario icl_det_gate_seed8 --mode det --graph-condition graph_ab \
  --request-profile gemma_e4b --provider dry-run --reasoning-mode off \
  --max-tokens 8192 --out /tmp/icl_graph_dry --tag graph_ab
```

For the additional worlds, use `--seed 13` or `--seed 25` with the preview
script. Add `--eligibility-report /tmp/icl_graph_eligibility.json` to save the
existing deterministic search over seeds 1 through 1000, including checks and
rejection reasons through seed 25. No selection criteria are added.
The scenario names are `icl_det_gate_seed13` and `icl_det_gate_seed25`.
Start, goal and queried pairs are derived from each world: the changed action
plus four unchanged controls, using the existing deterministic selection.

The planned local OFF extension is two worlds, each with all three conditions
and three repeated conversations, using sampling seeds 0, 1 and 2. Run each
block once under a new tag with the reviewed local configuration, keeping
K=budget=10, evidence seed 0 and output allowance 8192. Dry previews establish
reference correctness, not deployment readiness or model performance.
ON remains conditional on an imperfect, operationally valid OFF block and
uses all three repeats for that same graph and condition. Its OFF reference
must match the implementation commit, static prompts and deployment controls.

## Single-model handoff

After the reviewed implementation is committed and published, Maciej can fetch
`origin/feature/icl-graph-availability` and check out the shared full commit SHA
in a clean checkout. Confirm `git rev-parse HEAD` matches that SHA before running;
do not substitute current main or the older experiment commit.
Use the full implementation commit SHA supplied with the handoff.

Use Python 3.11 or 3.12, with the standard library only; no package installation
or credentials are needed for this offline check. From the repository root, use
an unused output directory:

```sh
python3 -B run_pilot.py --protocol icl_graph_availability_v1 \
  --scenario icl_det_gate_seed8 --mode det --graph-condition graph_ab \
  --request-profile sol --provider dry-run --reasoning-mode off \
  --repeats 3 --sampling-seeds 0 1 2 --max-tokens 8192 \
  --out /tmp/icl_graph_sol_handoff --tag graph_ab_off
```

The first planned block is Sol, `graph_ab`, reasoning OFF, graph seed 8:
three repeated conversations with two turns each. The dry run creates six
synthetic answers, not model results or readiness evidence. Sol does not send
sampling seeds; 0, 1 and 2 identify the repeated outputs. Use this single-model
entry point; the three-model launcher is not required.

Maciej reports Sol access, but a real launch command requires his exact provider,
model version and deployment identifier, endpoint/API version, authentication
method, supported OFF control and verification evidence, and context/output
limits. The graph path currently uses OpenAI-compatible Chat Completions;
Azure and direct OpenAI connection settings are not interchangeable. Review any
provider-specific compatibility gap before launch. Keep credentials and private
deployment configuration outside Git; do not use synthetic test configurations.

Outputs are saved under `<out>/<tag>/`: one JSON per conversation and
`summary.json`. Each run retains `turns.A` and `turns.B`, including prompts,
`raw_response` saved before parsing, `parsed`, `scored`, `provider_usage`,
recorded reasoning and, for real calls, the complete sanitized provider envelope.
The default output root is `pilot_artifacts`; preserve incomplete runs too.

## Real-run readiness

Real runs require a clean implementation commit, an unused output directory,
and a reviewed non-secret `--deployment-config` JSON. Set `--model` and
`--base-url` to exactly its verified model/deployment and endpoint. No provider
fallback, automatic resume, network retry, repair or history truncation is used.
Legacy sampling/control CLI flags are rejected for this protocol; profiles
construct the complete request without accepting arbitrary overrides.

The config records `profile`, `model_id`, `model`, `response_model`, `model_source`,
`endpoint`, `api_version`, `supported_request_fields`, `seed_supported`,
`effective`, `context`, `preflight` and `pricing`. See
[`validate_deployment`](../icl_graph.py) and the explicitly synthetic fixtures
in [the tests](../test_icl_graph.py) for the schema. Synthetic fixtures are not
readiness evidence and must not be used for a real run.

- Gemma profiles request temperature 1, top_p .95, top_k 64, max_tokens 8192
  and explicit reasoning off/on. Seeds are sent only with verified support.
  Effective repeat penalty is 1, Min P .05 where supported, with no separate
  reasoning budget or app response-length limit. Record the runtime,
  quantization, template hash and effective settings. Local E4B uses Q6_K,
  context 32768, batch 512/256, parallel 1, Flash Attention/KV offload on,
  speculation off. The 31B endpoint and its control support require confirmation.
- Together has an opt-in `gemma_31b_together` request profile for the exact
  `google/gemma-4-31B-it` model. It sends `reasoning: {"enabled": false/true}`
  and accepts either documented response reasoning field, rejecting conflicting
  aliases. Original envelopes remain intact. This adapter is not a readiness
  attestation: model-specific toggle support, effective settings and OFF
  evidence still need verification. Missing counts remain null/unknown,
  never zero. Existing request settings are unchanged.
- Sol uses Chat Completions with max_completion_tokens 8192 and
  reasoning_effort none/medium, with temperature/top_p/top_k/seed omitted in
  both modes. Provider sampling defaults remain unspecified. Use the actual
  model returned by the endpoint; no Azure deployment name is inferred.
  Sol's private template hash is unavailable: record null and a template source,
  not a fabricated hash. Local Gemma requires its actual template SHA-256.
- `preflight` contains the fixed request, complete sanitized response,
  HTTP status, empty retry list and evidence source. The control is checked,
  not mathematical correctness. OFF needs zero reasoning tokens and no
  substantive reasoning, or a documented empty channel and verified OFF
  template. ON needs positive evidence. Empty thought delimiters do not count.
- Hosted profiles use `hosted_readiness_v1`. `effective.hosted_evidence`
  must identify `policy`, `scope: exact_model`, `model`, `endpoint`, `date`,
  `source`, exact reasoning `request_fields` and `effect`:
  `disables_reasoning_computation` for OFF or `enables_reasoning` for ON.
  This records operator-supplied evidence, not independent verification of a
  provider's internals. Generic hybrid-model documentation is insufficient.
  OFF records `measured_zero` when the documented count is zero with no
  positive reasoning, requiring `reasoning_tokens_source`. If counts are
  unavailable, exact-model disable assurance can support
  `provider_documented_disable`; the count remains null and direct evidence
  unknown. Empty text alone never establishes OFF. Positive reasoning or a
  contrary control echo fails OFF. Optional documented `control_echo` specifies
  `path`, `expected` and `source`; any exposed value must agree.
- Hosted runtime/template metadata may be null only with `runtime_source` or
  `template_source` explaining unavailability. Inapplicable hosted app toggles
  need explicit `app_settings_inapplicable` reasons. Actual sampling controls
  remain mandatory, as do `output_context_source` and
  `no_reasoning_budget_source`. No inaccessible detail is guessed. The policy,
  preflight verification basis and complete deployment provenance enter run
  identity. Preflight, saved turns and independent raw-response audit use the
  same rule. Local E4B reasoning requirements remain unchanged.
- Hosted `context` requires `method: utf8_upper_bound`, `tokens`, a documented
  `overhead_tokens_per_message`, `replay_expansion_bound` and `source`.
  These are conservative bounds, not exact tokenizer counts. They must be
  justified for the effective tokenizer/template, not copied from estimates.
  Planning reserves the complete A output and B output; every actual B history
  is checked again. Insufficient or unverified bounds block generation.
- Local E4B uses `method: local_backend_tokenizer_v1` and conditional admission,
  not an invented reserve for an unseen answer. Before each turn, the active
  backend renders the complete messages with its verified template, generation
  prefix and thinking setting, then tokenizes with special tokens. Admission
  requires the saved count plus 8192 output tokens to fit 32768. The exact raw A
  answer is supplied before B, including malformed text. No history is repaired
  or truncated by the runner. Existing template formatting remains unchanged.
  Readiness must identify the backend, GGUF tokenizer, template, Bionic request
  code, app settings and loaded instance, and include a saved-preflight count
  matching reported prompt usage plus the generation-path verification source.
  Counts, rendered text, token IDs, settings and request/message hashes are saved
  for offline audit. The audit checks saved evidence and provider usage; it does
  not pretend to rerun an unavailable tokenizer. A discrepancy fails operations.
  Counting errors, deployment drift or insufficient capacity stop before the
  affected generation call and preserve any completed A as an incomplete run.
  Backend authentication comes only from `ECPM_LOCAL_BACKEND_API_KEY`, configured
  by the operator. The runner never discovers credentials from process arguments.
- `pricing` may supply input/output USD per million with a source. Costs are
  estimated without cache discounts; absent prices remain unavailable, while
  local runs are unpriced. Usage and complete sanitized provider envelopes are
  saved before answer parsing. Provider/model and control mismatches pause the
  run without replacing responses.

Run one condition at a time, easiest first. After all three OFF repeats for
one model/condition, `on_repeats` returns all three repeat indices if any saved
per-turn `correct` flag is false; otherwise it returns none. It refuses incomplete
or operationally failed OFF input. It never launches ON. Conditional ON is a
diagnostic, not a balanced reasoning-effect comparison.
Real ON execution requires `--off-reference` pointing to that completed OFF
directory, and checks matching commit, prompts, model and non-reasoning settings.

Seed 8 is the original development graph; seeds 13 and 25 are additional
eligible worlds, not new model results. Repeats within a graph are not
independent graph samples. Detection cannot measure false alarms here.
Five queried pairs do not measure full graph learning.
Supplying the graph adds explicit information and a shorter representation.
Difficulty ordering and separate internal abilities are not established.

Official references: [Sol model and pricing](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[Gemma thinking templates](https://ai.google.dev/gemma/docs/capabilities/thinking).
These documents do not establish account access or effective deployment settings.
