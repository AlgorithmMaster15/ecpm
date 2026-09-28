# Model first, direct tasks, or supplied graph

`icl_model_first_v1` is a separate, OFF-only protocol. Historical protocols,
prompts, parsers, scores and results are unchanged. The locked source package is
dated 28 September 2026. This implementation needs review and a committed code
identity before any real run; offline success is not deployment readiness.

The design bundle's generator, parser and baseline source copies each have one
extra trailing LF relative to the published files. The preview report records
both SHA-256 identities and verifies the exact one-byte relationship in memory.
Published files are unchanged; no source replacement or normalization is needed.

| Condition | Main workflow in each period | Separate measurement |
| --- | --- | --- |
| `model_first` | Logs, freely written model, tasks | All 16 transitions |
| `task_only` | Logs, tasks | All 16 transitions |
| `graph_given` | Logs and current graph, tasks | All 16 transitions |

The supplied B graph is updated. All conditions share observation order, menus,
mechanics and four deterministic route queries. The first query is the original
start/goal; three others are selected using only A reachability and the locked
hash ranking. Explicit unreachable answers remain part of the task.

Each measurement runs after its task but forks from the pre-task snapshot. It
excludes the current task question/answer. Measurement answers never enter main
history. B retains complete A model/task answers, including malformed text, but
never provider-private reasoning. History IDs and hashes are saved per request.

Model-first allows 4096 tokens for each model answer and 4096 for each task.
Other conditions allow 8192 for tasks. Every measurement allows 4096. Unused
allowance does not transfer. Actual full histories are checked before every
request with that stage's allowance; nothing is truncated. Operational failure
stops the block without retry. Normally finished wrong/malformed answers remain
results and do not stop later stages.

## Offline checks and commands

Use Python 3.11 or 3.12, standard library only. No dependencies, credentials or
model access are needed. Run from the repository root, using unused output paths:

```sh
python3 -B test_icl_model_first_contract.py
python3 -B test_icl_model_first.py
python3 -B experiments/preview_icl_model_first.py --out /tmp/icl_model_first_review
python3 -B run_pilot.py --protocol icl_model_first_v1 \
  --scenario icl_det_gate_seed8 --mode det \
  --model-first-condition model_first --request-profile sol \
  --provider dry-run --reasoning-mode off --max-tokens 8192 \
  --repeats 3 --sampling-seeds 0 1 2 \
  --out /tmp/icl_model_first_dry --tag sol_seed8_model_first_off
```

Repeat the dry command with `task_only` and `graph_given`, each using a fresh
tag. The three blocks contain nine synthetic conversations and 42 requests,
not model results. [The preview/report script](../experiments/preview_icl_model_first.py)
generates all 42 prompt files across seeds 8, 13 and 25, references, source-hash
reconciliation, and the run plan. Plan figures are generated, not usage forecasts.
Only seed 8 is proposed for the first authorized pilot. Seeds 13/25 need later
review and authorization; there is no automatic ON follow-up.

## Scoring and reporting

The new strict parser accepts exactly one JSON object, without fences/prose,
duplicate member names or a later-object rescue. Missing or duplicated pairs
fail their own row. Missing B `changed` fails joint scoring and Preservation,
not independently valid transition fields. Unexpected fields are format errors;
required values are retained. Invalid rows fail the all-16 denominator.
Extra route-step and localization fields likewise fail format without erasing
valid required values. An extra JSON field is not an extra route action: extra
goal actions, missing fields and invalid action choices still fail.
Destination accuracy and MAE report conditional scored counts. Null numerical
values do not acquire an invented MAE. Every route is parsed and graded
independently at its own query goal. Correct no-route answers are solutions,
not valid finite routes; invalid/unreachable cost and regret are N/A.

Self Preservation compares the independently elicited A/B control reports and
requires B `changed=false`. Truth Preservation compares B with truth without
requiring A correctness. Both show all-control and conditional denominators and
all-four-correct rates. Route/report consistency is diagnostic; own-report
optimality requires a complete valid report. Anchor and sampled routes remain
separate.

Generate a report from saved scores without reparsing or rescoring answers:

```sh
python3 -B experiments/preview_icl_model_first.py \
  --summarize /path/to/block1 /path/to/block2 /path/to/block3 \
  --out /path/to/unused_results.json
```

Every report recomputes the operational audit. A completed record that fails it
is listed as quarantined, with no scientific scores or denominator contribution.
Operationally valid wrong or malformed answers remain in all-response counts.
Incomplete attempts are listed separately. Compatible groups require the same
exact model/provider, synthetic/live status, code/lock/prompts/scorer, world,
queries and deployment/control identity; only the repeat and its requested seed
may vary. Different identities are separated, never silently pooled. Duplicate
repeats within a compatible group are rejected even if run IDs differ, and only
repeat labels 0/1/2 are allowed. Preserve original artifacts and saved scores.

## Human review of freely written models

Explicit-model scores are pending until Pavlos and Maciej have independently
extracted the original visible model texts, before inspecting oracle scores.
They are N/A for conditions that were not asked to construct a model. Do not
infer a correct free-form model from a correct structured measurement.

```sh
python3 -B experiments/preview_icl_model_first.py \
  --manual-template /path/to/model_first_run.json --reviewer Pavlos \
  --out /path/to/unused_pavlos_form.json
python3 -B experiments/preview_icl_model_first.py \
  --manual-import /path/to/model_first_run.json \
  --reviews /path/to/pavlos_form.json /path/to/maciej_form.json \
  --out /path/to/unused_manual_results.json
```

Every claim needs a source hash, quote and zero-based `[start,end)` span.
Enter explicit values only; literal always/never may map to 1/0 and explicit
fractions to their numerical value. Do not fill gaps using logs or truth, execute
generated code, or use a model judge. Availability needs an explicit statement
or scoped rule. B inheritance needs both the B rule and A claim citations.
Conflicts and unstated facts remain unknown. Mark each independent form complete
only after review. Optional `--resolutions` selects a reviewer's cited claim or
`unknown`, with a note, for each disputed `period/state:action/field`. Original
reviews and disagreements remain recorded. The import validator checks spans
and values, not semantic entailment. Unreviewed work is never scored as zero.

## Maciej handoff: real-run prerequisites

After review, use the supplied full implementation commit in a clean checkout.
Do not substitute the older graph-availability commit or current main. Run the
offline commands above first. One model can run independently; no three-model
launcher is required.

Sol uses the existing `sol` Chat Completions profile: `reasoning_effort=none`,
stage-specific `max_completion_tokens`, no temperature/top_p/top_k/seed. Labels
0/1/2 identify repeated outputs, not independently seeded Sol samples. Local
Gemma uses its reviewed OFF profile, with verified seed support only. No exact
Sol provider, endpoint, deployment ID or credentials are inferred here.

A private `--deployment-config` must contain `deployment`, the existing verified
[graph-readiness configuration](ICL_GRAPH.md), and `model_first`:

```json
{
  "protocol": "icl_model_first_v1",
  "output_allowances": [4096, 8192],
  "stage_limits_source": "<evidence both request limits reach the actual backend>",
  "system_message_source": "<evidence the system role uses the intended generation path>",
  "context_source": "<evidence actual multi-turn template/counting or hosted bound is valid>"
}
```

This is the shape of `model_first`, not a ready configuration. These sources must
be real evidence, not copied test fixtures. Existing preflight/control evidence
may be reused only if applicable. The new system-role/multi-stage path and both
output limits need confirmation. Hosted unavailable reasoning counts remain
null; explicit exact-model assurance is required for the documented-disable
path. Actual input and prices remain unknown until verified. Input plus each
stage's allowance must fit; an earlier short answer cannot bound future answers.

Once the code is reviewed/committed, deployment evidence is accepted and runs
are separately authorized, the actual single-block CLI is:

```sh
python3 -B run_pilot.py --protocol icl_model_first_v1 \
  --scenario icl_det_gate_seed8 --mode det \
  --model-first-condition model_first --request-profile sol \
  --provider openai --model '<VERIFIED_MODEL_OR_DEPLOYMENT_ID>' \
  --base-url '<VERIFIED_COMPATIBLE_ENDPOINT>' \
  --deployment-config '<PRIVATE_VERIFIED_MODEL_FIRST_CONFIG.json>' \
  --reasoning-mode off --max-tokens 8192 --timeout 900 \
  --repeats 3 --sampling-seeds 0 1 2 \
  --out '<NEW_OUTPUT_ROOT>' --tag sol_seed8_model_first_off_v1
```

Then use `task_only` with tag `sol_seed8_task_only_off_v1`, followed by
`graph_given` with tag `sol_seed8_graph_given_off_v1`, only after each preceding
block passes its operational audit. No command is executed by this handoff.
`openai` names the adapter, not an assertion that Azure and direct OpenAI have
identical endpoint/authentication semantics. Confirm compatibility before launch.
Hosted profiles read `OPENAI_API_KEY`, or `TOGETHER_API_KEY` for Together. Local
Gemma never reads or forwards either hosted key. Its loopback frontend uses no
Authorization unless the operator separately configures `ECPM_LOCAL_API_KEY`.
`ECPM_LOCAL_BACKEND_API_KEY` remains limited to the tokenizer/backend checks and
is never used for generation. Credentials never belong in command arguments,
deployment JSON or review packages. Do not copy hosted keys into local variables.

Return every run JSON, summary, actual command, implementation/prompt hashes,
sanitized deployment/control/context evidence, and human extraction forms when
completed. Each run saves raw provider envelopes before answer parsing, original
visible model/task/readout text, parsed fields, scores and per-stage usage/cost.
Keep main and measurement cost separate. Do not upload credentials, private
machine paths or bulk historical evidence to Git.

This is a workflow comparison, not proof of an internal world model. Allowed
output is matched; actual computation, input, latency and calls are not. Repeats
within a graph are not independent worlds. A negative result remains a result.
