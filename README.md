# FedFisher: Federated Effect-Modifier Research

This repository studies treatment-effect-modifier discovery with binary and
survival outcomes, building on the original FedFisher one-shot federated-learning
code. The paper aims to combine evidence for efficient discovery when hospital
records cannot be pooled and FedFisher fitting uses one communication round,
with substantive effect-modifier findings in real EHR data. The implemented
simulation workflows support that research; they do not constitute a completed
EHR study or an end-to-end one-round EHR analysis.

The original binary classification, supervised signal-recovery experiments and
numeric CSV loader remain available. The original image datasets and image
models have been removed.

## Project navigation

| Work on | Start here |
| --- | --- |
| Code structure and dependency rules | [Architecture](docs/architecture.md) |
| Development and verification | [Development guide](docs/development.md) and [test guide](tests/README.md) |
| Original binary/base pipeline | [SyntheticBinary](synthetic_binary_experiment/README.md) and [complete original reference](docs/legacy/root-readme.md) |
| Binary effect modifiers and factorial variants | [Effect-modifier guide](effect_modifier_experiment/README.md) |
| Multiround FedAvg and homogeneous R methods | [Causal comparison](effect_modifier_experiment/causal_comparison/README.md) |
| FedFisher-EM, DR-Sieve and SCMV2 | [One-shot effects](effect_modifier_experiment/one_shot_effects/README.md) |
| Saved-model gradient and approximation diagnostics | [Diagnostics](effect_modifier_experiment/one_shot_effects/diagnostics/README.md) |
| Original Diag/KFAC CPU timing | [Timing](effect_modifier_experiment/fedfisher_timing/README.md) |
| Small models, signal strengths and recovery | [Model/signal ablations](effect_modifier_experiment/model_signal_ablation/README.md) |
| Survival methods and recovery | [Survival](effect_modifier_experiment/survival/README.md) |
| MLP8 survival prediction and modifier discovery | [Federated and pooled MLP8 comparison](effect_modifier_experiment/survival_mlp8/README.md) |
| Ten-seed survival method comparison | [FedAvg and accepted-model comparators](effect_modifier_experiment/survival_comparators/README.md) |
| Historical pooled survival controls | [Pooled probability benchmark](effect_modifier_experiment/survival_pooled/README.md) |
| Shared timing, fingerprints and recovery arithmetic | [Components](effect_modifier_experiment/components/README.md) |
| Artifacts, provenance and read-only evidence | [Infrastructure](effect_modifier_experiment/infrastructure/README.md) |
| Scheduler validation, dispatch and publication | [Scheduling](effect_modifier_experiment/scheduling/README.md) |
| Task schemas, disabled proposals and examples | [Scheduler contracts](effect_modifier_experiment/server_scheduler/README.md) |
| Tables and figures from completed artifacts | [Reporting](reporting/README.md) |
| Compatible commands and historical export files | [Scripts](scripts/README.md) |

## Execution and storage

The canonical source tree is `/home/del6500/projects/FedFisher`. Keep source and
small metadata here, durable datasets/models/results under
`/data/del6500/FedFisher`, and disposable caches, environment files and temporary
outputs under `/scr/del6500/FedFisher`.

The GitHub source tree contains code, documentation, dependency specifications
and small reviewed fixtures and metadata. Datasets, checkpoints, experiment
outputs, environments and caches stay in their designated data or scratch
trees. Existing historical results and source-bound proofs remain preserved;
see the [publication review workflow](docs/development.md#prepare-a-github-push).

New scientific work requires a centrally registered ServerScheduler task. The
foreground entrypoint remains
[scripts/fedfisher_server_scheduler_adapter.py](scripts/fedfisher_server_scheduler_adapter.py).
An outbox file or disabled proposal is not a submitted or admitted job.
Deterministic foreground reporting has the limited exception defined by the
root protocol. Historical Screen, resource-lock and Slurm commands remain audit
or export references.

Follow [AGENTS.md](AGENTS.md), the [project context](docs/codex/PROJECT_CONTEXT.md)
and the [scheduler runbook](docs/codex/SCHEDULER_RUNBOOK.md). Changing evidence
belongs in the [handoff](docs/codex/SCHEDULER_HANDOFF.md); Git, scientific manifests
and read-only central state establish current status. Preserved results and
proofs retain their original source bindings.

## Development

The pinned scientific dependencies remain in [requirements.txt](requirements.txt).
The commands below describe the configured lab host and its canonical checkout;
this repository is not a standalone installed distribution. Use the existing
science environment for bounded development fixtures:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py
```

The runner limits numerical libraries to one thread and writes caches and
temporary files to project scratch. See the [development guide](docs/development.md)
for optional reporting dependencies, focused checks and scientific release gates.
