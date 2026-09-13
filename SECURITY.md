# Security policy

## What this repository is

Research code for a paper. There is no deployed service, no user data, and no
released package. It is pure-stdlib Python that generates graphs, builds
prompts, calls model APIs when given credentials, and scores the replies.

The realistic risks are narrow, so this policy is narrow.

## Reporting a vulnerability

Email bektes23@itu.edu.tr rather than opening a public issue. Expect a reply
within a week. There is no bounty and no formal disclosure timeline.

## What is worth reporting

**A credential leak.** The harness reads `AZURE_OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` from the environment and must never write them into an
artifact. Every run artifact records the endpoint and model, deliberately, but
never the key. If you find a key, an endpoint with embedded credentials, or
anything that looks like one in `runs/`, in a notebook, or in git history,
report it privately and do not open an issue quoting it.

**An evaluator leak.** `prompt_view()` is the boundary between what a model may
see and what only the scorer may see. `assert_prompt_safe()` in
`ecpm_baseline.py` enforces the same boundary for the null model. A path that
puts `counts_*`, `world_*`, `change`, `oracle` or `seeds` into a prompt is a
correctness failure serious enough to invalidate results, so treat it like a
vulnerability: report it, and expect affected artifacts to be regenerated.

**Anything that executes untrusted input.** The tree is stdlib only and does
not evaluate model output as code. If that changes, it is worth a look.

## What is not a vulnerability

A model answering badly. A scorer disagreeing with your expectation. A run
costing more than expected. Those are issues or pull requests.

## What is switched on

| feature | state | note |
| --- | --- | --- |
| Secret scanning | on | backs up the credential rule above: a pushed key is flagged |
| Security advisories | on | |
| Code scanning | on | CodeQL, `.github/workflows/codeql.yml`, weekly and on pull requests |
| Dependabot | actions only | `.github/dependabot.yml`. The tree is stdlib only, so the workflows are the only dependencies |
| Private vulnerability reporting | see below | |

`.github/workflows/` carries eight workflows and no more. Scanners for
languages and artifacts this repository does not contain were deleted rather
than disabled: there is no Ruby, PHP, Java, Rust, Kotlin, Clojure, Elixir, R,
Terraform, Kubernetes manifest, Dockerfile, container, mobile app or deployed
service here. Neither were the ones needing a paid account kept, because a
scanner that fails on every push teaches people to ignore the Actions tab.

Adding one back is deliberate work, not a default. If a tool is worth running
it is worth a commit explaining what it covers that codeql and bandit do not.

## Bandit

Configured by `.bandit` at the repository root. Untuned it reports 413
findings, 378 of them `assert_used` in the test suites, which is the tool not
knowing which files are tests rather than a finding. Tuned, it reports 12.

Every skip has a written reason in that file, and three checks are
deliberately left visible: `B310` on the eight `urlopen` call sites that build
URLs from `--azure-endpoint` and `--base-url`, plus `B112` and `B104`. Those
are benign while the user is the operator, and they are exactly where it would
matter if that ever stopped being true.

`devskim` was removed. It reported 185 findings on a Python tree and is aimed
at C and C++ idioms; keeping it was an error in the earlier prune, which had
already classified it as not applicable before keeping it anyway.

## Scorecard findings

`scorecard.yml` rates this repository against OSSF supply-chain criteria
written for widely depended-upon open source. Some apply here and some do
not, so this records which is which rather than leaving the Security tab
ambiguous.

**Fixed.** Token-Permissions: every workflow now declares least privilege at
the top and widens it per job only where needed. Without that block a
workflow inherits the repository default, which can be write-all.

**Worth doing, a settings change rather than a file.** Branch-Protection.
Nothing currently stops a direct push to `main`, and several went in during
the September integration. Requiring the `tests` check and one review would
have caught at least one stale-base patch.

**Open.** Pinned-Dependencies. Thirteen actions are pinned by tag rather than
by commit SHA, so a tag could be moved under us. Pinning by SHA is correct and
Dependabot maintains SHA pins, but it is a mechanical change worth doing in
one pass rather than piecemeal.

**Addressed differently.** Fuzzing. The check asks for OSS-Fuzz or
ClusterFuzzLite, which is heavy for a repository this size, but the concern
behind it is real: `ecpm_parser.py` reads model output, which is
unconstrained text, and a crash there loses an instance where a wrong answer
would merely score badly.

`test_ecpm_parser.py` now fuzzes all four probes with 2,400 seeded replies
per run, half random bytes and half structurally valid JSON of the wrong
shape, and requires every one to return a status rather than raise. Seeded,
so a failure reproduces. Verified by injecting a defect into `run_probe` and
confirming the test fails.

The Scorecard alert will stay open, because it looks for a specific
integration rather than for the property. That is a reasonable thing to
leave open with a written reason.

**Does not apply.** Maintained: the check measures commit frequency over a
90-day window and this repository is younger than that. It resolves with
time, not with a change. Code-Review: the project pushes to `main` by design at this
stage, which the Branch-Protection item above would change if the group wants
it. Packaging and Signed-Releases: nothing is released.

Private vulnerability reporting is a repository setting rather than a file. If
it is enabled, use it in preference to email, because it opens a private
thread on the repository instead of relying on one inbox. If it is not, email
is the route.

## Supported versions

Only `main`. Tagged snapshots such as `v2.1-prefreeze` exist for provenance and
are not maintained.
