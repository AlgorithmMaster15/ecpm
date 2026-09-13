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

Link an issue with a closing keyword so it closes on merge: `Closes #12`,
`Fixes #12`. See
https://docs.github.com/articles/closing-issues-using-keywords

Every pull request carries at least one label, and CI refuses one that does
not. Most are applied automatically from the paths you changed: `env`,
`freeze`, `arm/icl`, `arm/agentic`, `arm/finetuning`, `arm/baseline`, `docs`,
`chore`.

Four are always added by hand, because path matching cannot tell:

- `needs-rerun` the change invalidates existing results. Name the affected
  artifacts.
- `spends-credit` it consumes API budget. Say how much.
- `provisional` a claim without a script or a written definition behind it.
- `blocked` waiting on someone else, named in the thread.

The set lives in `.github/labels.yml` and is synced from there, so add a label
by editing that file rather than through the web UI.

## Milestones

One milestone per real deadline, named for the deadline rather than for a
phase. A milestone without a date is a label wearing a different hat.

An issue belongs to a milestone only if missing that date would matter.
Everything else stays unassigned, which keeps the milestone readable as a
list of what genuinely has to happen by then.

## Issues

Open questions belong in issues, not in prose. A question recorded in a
document is found by whoever reads that document; a question recorded as an
issue is found by whoever is looking for work.

That applies particularly to the two kinds this project keeps producing: a
number with no script behind it, and a decision nobody has made. Both have a
template.

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

## Formatting

`.editorconfig` and `.gitattributes` handle line endings and indentation. LF
everywhere except `.ps1`, final newline, no trailing whitespace outside
markdown.

This is not cosmetic. `test_resource_mdp.py` asserts the shipped example
records regenerate byte-identically, so a checkout that converted them to
CRLF fails that test for a reason nobody would guess, and a `.sh` file with
CRLF does not execute on Linux at all.

There is deliberately no line-length rule: fifteen files exceed 79 columns
and the longest line is 166. A rule the tree breaks on the day it lands
teaches people to switch the tool off.

## Dependencies

Stdlib only. CI fails if that stops being true, which is the intended alarm
rather than an inconvenience.
