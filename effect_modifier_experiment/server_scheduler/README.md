# Scheduler contract documents

This directory contains strict scientific parameter schemas, disabled project
proposals, fixed plans and historical migration records. It does not
establish current central registration, submission or scientific completion.
Validation and dispatch are in [scheduling](../scheduling/README.md); deterministic
request specifications belong to the experiment packages.

Checked-in request examples live with the tests in
[`tests/fixtures/server_scheduler_requests/`](../../tests/fixtures/server_scheduler_requests/).
They illustrate syntax and exercise contract validation; they are not outbox
deliveries or submitted jobs. Running-job fixtures remain separately in
[`tests/fixtures/server_scheduler/`](../../tests/fixtures/server_scheduler/).

| Contract family | Documents here | Request fixtures |
| --- | --- | --- |
| Original CPU LASSO | `registration-proposal.toml`, `*-shard-parameters.schema.json` | [original-lasso](../../tests/fixtures/server_scheduler_requests/original-lasso/) |
| Causal comparison | `comparison-registration-proposal.toml`, `comparison-*-parameters.schema.json` | [comparison](../../tests/fixtures/server_scheduler_requests/comparison/) |
| Additional one-shot effects | `one-shot-effects-v1/` | [one-shot-effects-v1](../../tests/fixtures/server_scheduler_requests/one-shot-effects-v1/) |
| Saved-model diagnostics | `effects-replay-v1/` | [effects-replay-v1](../../tests/fixtures/server_scheduler_requests/effects-replay-v1/) |
| Original Fisher timing | `fedfisher-timing-v1/` | [fedfisher-timing-v1](../../tests/fixtures/server_scheduler_requests/fedfisher-timing-v1/) |
| Model/signal ablation | `model-signal-ablation-v1/` | [model-signal-ablation-v1](../../tests/fixtures/server_scheduler_requests/model-signal-ablation-v1/) |
| Model/signal recovery | `model-signal-recovery-v2/` | [model-signal-recovery-v2](../../tests/fixtures/server_scheduler_requests/model-signal-recovery-v2/) |
| Archived failed-shard completion | `model-signal-completion-v3/` | [model-signal-completion-v3](../../tests/fixtures/server_scheduler_requests/model-signal-completion-v3/) |
| Survival | `survival-v1/` | [pilot](../../tests/fixtures/server_scheduler_requests/survival-v1/) and [formal](../../tests/fixtures/server_scheduler_requests/survival-v1/formal/) |
| Survival recovery | `survival-recovery-v1/` | [survival-recovery-v1](../../tests/fixtures/server_scheduler_requests/survival-recovery-v1/) |
| Survival repair | `survival-repair-v2/` | [survival-repair-v2](../../tests/fixtures/server_scheduler_requests/survival-repair-v2/) |
| Survival integrated epoch | `survival-integrated-v3/` | [survival-integrated-v3](../../tests/fixtures/server_scheduler_requests/survival-integrated-v3/) |
| MLP8 prediction | `survival-mlp8-v1/` | [survival-mlp8-v1](../../tests/fixtures/server_scheduler_requests/survival-mlp8-v1/) |
| MLP8 discovery | `survival-mlp8-v2/` | [survival-mlp8-v2](../../tests/fixtures/server_scheduler_requests/survival-mlp8-v2/) |
| Historical pooled controls | `survival-pooled-v1/` | Running fixtures and request-builder tests |
| Ten-seed survival comparators | `survival-comparators-v1/` | Running fixtures and request-builder tests |

Normal requests express scientific work and omit machine/profile choices.
Publication goes only to the FedFisher scratch outbox. An operator validates
and submits requests through the central scheduler. The foreground command
path remains unchanged. Follow the [runbook](../../docs/codex/SCHEDULER_RUNBOOK.md)
and preserve the [legacy-results policy](legacy-results-policy.md) and
[historical migration inventory](migration-inventory.md).
