# Exploratory probability-elicitation pilot (seed-7 pair)

Runs on top of the signed-off tree. Nothing in v2.1-prefreeze,
Section 7, or schema 2.1 changes; the route half is scored by the
frozen grader itself.

## Files

- packet_det_model.json, packet_sto_model.json
  Prompt-safe: the exact prompt_view payload (F2_shuffled, both
  periods, k = B = 5) plus the 15 listed pairs. These may be pasted
  to a model.
- packet_det_oracle.json, packet_sto_oracle.json
  Evaluator only, never paste: true probabilities pre/post, empirical
  rates (both prompt-visible and full-collection), the change record,
  and the route oracle.
- score_probability_pilot.py
  Place in the repo root at 5318c3e and run:
      python3 score_probability_pilot.py --mode sto response.json

## Response format the prompt must request

    {"estimates": [{"node": "A", "action": "a1",
                    "p_pre": 0.8, "p_post": 0.8},
                   ... every listed pair exactly once ...],
     "route": [{"node": "E", "action": "a2"}, ...]}

Suggested route sentence for the prompt (field-tests the reworded
ask from pilot finding 2): "at most 32 steps; the first step's node
must be {start}; each next step's node must be where the previous
action leads; the destination of the last action must be {goal};
do not include a step at {goal}."

## What the scorer reports

Coverage (all 15 pairs exactly once), MAE of p_pre/p_post against the
true probabilities and against the prompt-visible rates, the model's
numbers for the changed pair, an elicited localization (the pair with
the largest |p_post - p_pre| gap), and the route scored by the frozen
Section 7 grader (status, expected cost, regret).

## Calibration anchors (from self-test, stochastic instance)

- A response that reproduces the prompt-visible rates exactly:
  MAE vs truth 0.171 pre, 0.167 post, 0.169 overall; elicited
  localization is a tie between B a1 and E a1, both wrong (the scorer
  reports the full tied set).
- A response with the true probabilities: MAE 0.169 vs the visible
  rates, localization D a2 (correct), route regret 0.
Read model numbers against these two anchors: at k = 5, perfect
evidence reading and truth sit about 0.17 apart, and perfect reading
still localizes to noise pairs.

## Provenance

For every run save the exact prompt, raw response, displayed model and
version, date, and settings. One instance per mode: report strictly as
a sanity pilot, not a model comparison.
