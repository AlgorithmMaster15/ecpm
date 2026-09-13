# Contributing

Conventions already in use, written down so they survive.

`AGENTS.md` carries the same rules in short form, for AI coding agents. It is
deliberately brief, because agents load it on every prompt and length there
costs context that would otherwise go to the code. When a convention changes,
change both.

## The freeze

Schema 2.1 is frozen at `5318c3e`. Artifacts are valid only if produced
against that environment tree, and each records `frozen_sha`, `git_head` and
`pinned_to_freeze`.

Changes above the freeze are allowed if they are additive. `redirect` is the
worked example: it emits its second endpoint as `change.new_edge` only for
redirect instances, so records for the other five conditions are
byte-identical and the shipped examples still regenerate exactly.

`test_resource_mdp.py` checks that. If it reports the examples are stale, a
change was not additive.

## Branches and patches

Work on a branch, open a pull request, do not push to main directly.

Before starting, and before applying any patch:

    git status --short
    git branch --show-current

Both have caught real problems: an uncommitted file silently blocking a
patch, and a `git am` applying to the wrong branch.

After applying anything, check by content rather than by the absence of an
error. A patch can fail while git prints nothing useful, and a merge can
succeed while dropping a fix.

## Commit messages

Say what changed, why it was wrong before, and how it was verified.

Prefix by area where it helps: `env:`, `parser:`, `baseline:`, `harness:`,
`docs:`.

Describe the artifact, not the intent behind it.

## Tests

Six suites, stdlib only, run from the repository root:

    python3 test_resource_mdp.py
    python3 test_ecpm_parser.py
    python3 test_explore_agent.py
    python3 test_ecpm_baseline.py
    python3 test_prompt_contract.py
    python3 test_run_pilot.py

CI runs all six on push plus the dry-run pilots.

**A check that has only ever passed has been run, not tested.** When adding
one, damage a passing input and confirm the check fails, and says which
guard caught it. `test_ecpm_baseline.py` has an injected-defect test as the
worked example.

**Nothing examined must not score as a pass.** An empty input and a clean
result need different exit states. The baseline returns `could_not_run`
rather than `detection: false` for this reason. A test that skips a missing
fixture and still reports PASS is the same defect.

## Numbers

Every reported number needs a script that produces it and an artifact that
holds it. Tables in documents are generated, not transcribed.

Attach `n` to every cell. Cells with `n` in the tens carry an interval, not
a point claim.

Claims taken from a team document rather than from a script in this tree are
cited to that document and marked provisional until the definition behind
them is written down.

## Writing

No em dashes, in code, comments, commit messages or documents. Use a comma,
a colon, a full stop or a line break.

State what is not done as explicitly as what is. The README has a section
for it.

## Dependencies

Stdlib only. CI fails if that stops being true, which is the intended alarm
rather than an inconvenience.
