# ECPM

Can a frozen LLM infer an environment's transition structure from
observation logs alone, and replan after a hidden change?

The environment is an 8-node packet-routing MDP, generated as a pair:
M0, and a copy M1 with zero or one recorded intervention. The model sees
balanced evidence from both periods and answers four probes: detection,
localization, preservation, adaptation. Routes are scored by execution
in the true simulator (expected cost and regret), never by the model's
own judgment.

## Contents

- `resource_mdp.py`: environment: paired generator, evidence collection,
  `prompt_view()`, oracle
- `ecpm_parser.py`: frozen parser and probe scoring (INTERFACE.md §7)
- `test_resource_mdp.py`, `test_ecpm_parser.py`: tests, stdlib only
- `parser_fixtures.json`: format-level parser cases
- `adversarial_review.py`, `ecpm_reply_verification.py`: review tooling
  used for the freeze sign-off
- `example_deterministic_silent_break.json`,
  `example_stochastic_silent_break.json`: the matched seed-7 showcase pair
- `INTERFACE.md`: schema 2.1 and the frozen answer contract

## Freeze

Schema 2.1 is frozen at commit `5318c3e`. Pilot artifacts are valid only
if produced on that commit; each artifact records `pinned_to_freeze`.

## Quickstart

    git clone https://github.com/itu-itis25-bektes23/ecpm
    cd ecpm
    git checkout 5318c3e
    python3 test_resource_mdp.py
    python3 test_ecpm_parser.py

## Running the pilots

`run_pilot.py` (kept out of the frozen tree; joins at merge) runs both
seed-7 silent-break pilots end to end. Place it in the repo root on the
frozen commit:

    python3 run_pilot.py                # dry run, no API key needed

    AZURE_OPENAI_API_KEY=... python3 run_pilot.py \
        --provider azure --model YOUR-DEPLOYMENT \
        --azure-endpoint https://YOUR-RESOURCE.openai.azure.com \
        --max-tokens 4096

    ANTHROPIC_API_KEY=... python3 run_pilot.py \
        --provider anthropic --model claude-sonnet-4-6 --max-tokens 4096

The 1024 default for max_tokens truncates verbose models mid-answer;
pass --max-tokens 4096 until the default is raised at merge.

Outputs land in `pilot_artifacts/`. For review, send
`pilot_deterministic.json`, `pilot_stochastic.json`, and `run_pilot.py`.

## Azure access

1. $200 startup credits: https://www.microsoft.com/en-us/startups
   (sign in, create the Azure account, complete identity verification).
2. In portal.azure.com create an Azure OpenAI resource. The resource
   name sets the endpoint: `https://NAME.openai.azure.com`.
3. In the Foundry portal deploy a chat model (e.g. gpt-4o). The
   deployment name is the `--model` argument; keys are under
   Keys and Endpoint. If you hit a 404, check the deployment name and
   try `--api-version 2024-10-21`.

## Pilot status (23 Aug 2026)

Both seed-7 pilots have run on the frozen commit with three models:
claude-sonnet-4-6 (official run), gpt-4o (Azure), and Gemma 4 E4B
(local LM Studio, OpenAI-compatible endpoint). Artifacts are archived per run under `runs/` on main.
Findings and proposed merge fixes are in the team research doc
("The Gist", pilot results section).

## Write-up

Full explanation and figures: "ECPM Phase 2: The Gist" in the team
research doc.
