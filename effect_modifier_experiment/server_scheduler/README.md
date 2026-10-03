# Scheduler contract documents

This directory contains strict scientific parameter schemas, disabled project
proposals, request examples and historical migration records. It does not
establish current central registration, submission or scientific completion.
Validation and dispatch are in [scheduling](../scheduling/README.md); deterministic
request specifications belong to the experiment packages.

| Contract family | Documents |
| --- | --- |
| Original CPU LASSO | `registration-proposal.toml`, `*-shard-parameters.schema.json`, `examples/` |
| Causal comparison | `comparison-registration-proposal.toml`, `comparison-*-parameters.schema.json` |
| Additional one-shot effects | `one-shot-effects-v1/` |
| Saved-model diagnostics | `effects-replay-v1/` |
| Original Fisher timing | `fedfisher-timing-v1/` |
| Model/signal ablations and recovery | `model-signal-ablation-v1/`, `model-signal-recovery-v2/` |
| Survival and recovery epochs | `survival-v1/`, `survival-recovery-v1/`, `survival-repair-v2/`, `survival-integrated-v3/` |

Normal requests express scientific work and omit machine/profile choices.
Publication goes only to the FedFisher scratch outbox. An operator validates
and submits requests through the central scheduler. The foreground command
path remains unchanged. Follow the [runbook](../../docs/codex/SCHEDULER_RUNBOOK.md)
and preserve the [legacy-results policy](legacy-results-policy.md) and
[historical migration inventory](migration-inventory.md).
