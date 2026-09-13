# AGENTS.md

Research repository. An 8-node routing MDP, generated in pairs, used to test
whether a model notices a hidden change in observation logs.

`CONTRIBUTING.md` has the full conventions. This file is the short form.

## Setup

None. Stdlib only, Python 3.11 or 3.12. **Never add a dependency.** CI fails
if an import outside the stdlib appears, and that alarm is intentional.

## Test

Run from the repository root:

    python3 test_resource_mdp.py
    python3 test_ecpm_parser.py
    python3 test_explore_agent.py
    python3 test_ecpm_baseline.py
    python3 test_prompt_contract.py
    python3 test_run_pilot.py

Dry runs make no API calls and are safe:

    python3 -B run_pilot.py --condition redirect --seed 7 --mode sto --tag scratch

## Rules that are not obvious from the code

**Do not move or rename the eight root modules** (`resource_mdp`,
`ecpm_parser`, `ecpm_baseline`, `explore_agent`, `explore_metrics`,
`model_clients`, `prompts`, `run_pilot`). Contributors have open branches
that import them by name. A `src/` layout is planned and deliberately
deferred.

**The environment is frozen at `5318c3e`.** Changes above the freeze must be
additive: emit new fields only for the case that needs them, so existing
records stay byte-identical. `test_resource_mdp.py` fails if the shipped
examples no longer regenerate exactly. That failure means a change was not
additive, not that the examples are out of date.

**A check that has only ever passed has been run, not tested.** When adding
a check, damage a passing input and confirm the check fails and names the
guard that caught it. See the injected-defect test in
`test_ecpm_baseline.py`.

**Nothing examined must not score as a pass.** An empty input and a clean
result need different exit states. `ecpm_baseline.run_baseline` returns
`could_not_run`, not `detection: false`, for this reason. A test that skips
a missing fixture and still prints PASS has this defect.

**Numbers in documents are generated, not typed.** Every reported figure
needs a script in `experiments/` that produces it and an artifact in `runs/`
that holds it. Attach `n` to every cell.

**No em dashes** anywhere: code, comments, commit messages, documents. Use a
comma, a colon, a full stop or a line break.

## Pull requests

Branch, then open a PR. Do not push to `main`.

Before applying any patch, check `git status --short` and
`git branch --show-current`. After applying, verify by content rather than by
the absence of an error: patches fail quietly and merges can drop a fix while
still reporting success.

Commit messages say what changed, why it was wrong, and how it was verified.
Describe the artifact, not the intent behind it.
