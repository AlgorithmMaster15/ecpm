# Gist correction — sample size and uncertainty (v2.1, 2026-08-19)

## Delete

- The sentence claiming the run covers "**150 independent networks**".
- The "**±7 points**" worst-case CI figure.

## Replace with (paste-ready)

**Design and sample size.** Each cell of the design uses **30 graph
seeds**; the five conditions are generated **on the same 30 graphs**, so
one mode x format sweep yields 30 x 5 = 150 instances but only **30
independent networks** — five repeated measurements per graph. The graph
**seed is the unit of clustering**: condition contrasts are computed
**within seed** (paired), and uncertainty is reported with
**seed-clustered bootstrap CIs** (resample seeds, never instances).

For a single proportion the worst-case 95% CI half-width is about
**±17.9 pp at n = 30** (one cell) and **±8.0 pp at n = 150**; the ±8
figure applies only to claims that legitimately pool across conditions,
and the bootstrap must stay seed-clustered even then. The paired design
is the compensation: within-seed contrasts remove between-graph
variance, so condition *differences* are estimated with more power than
the raw n suggests.

Temperature-0 decoding is treated as deterministic per prompt, but a
small **repeat audit** (identical prompt, 3 runs per model) confirms
this; any residual nondeterminism is folded into the bootstrap.

## Wording note (review item 7)

Report metrics as **behavioral performance under each exposure format**
(detection rate, localization accuracy, regret), not as claims about
internal representations. "The model builds/does not build a world
model" is out of scope for Phase 2 data; "F3 supports detection where F1
does not" is in scope.
