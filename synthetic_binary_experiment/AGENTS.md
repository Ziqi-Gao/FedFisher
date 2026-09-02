# SyntheticBinary Agent Contract

This contract is additive to the repository-level `AGENTS.md`. It may not
weaken the root write-scope, ServerScheduler, provenance, or shared-compute
rules. Read `docs/codex/PROJECT_CONTEXT.md` before working in this subtree and
`docs/codex/SCHEDULER_RUNBOOK.md` for any execution-related work.

## Current execution policy

- `SyntheticBinary` has no registered ServerScheduler task. All new training,
  evaluation, summarization, plotting, sweeps, and other computation in this
  subtree are blocked until separately adapted, profiled, fixture-tested,
  piloted, and centrally registered.
- Direct Python commands, Slurm files, Screen/tmux workflows, and local launch
  scripts in this repository are preserved legacy or export surfaces. They do
  not authorize computation on the current server.
- Repository-local `synthetic_binary_experiment/outputs/` and `logs/` paths are
  historical only. A future registered task must store durable data and results
  under `/data/del6500/FedFisher` and regenerable logs, runtime state, and caches
  under `/scr/del6500/FedFisher`.
- Preserve historical source, results, manifests, and logs unless the user gives
  a separate, exact deletion instruction.

## Scope

- Keep the synthetic task on the original FedFisher code path.
- The synthetic adaptation adds data, model, evaluation, and helper-script entry
  points. It does not change FedFisher aggregation logic.
- Do not modify the original aggregation path unless the user explicitly asks:
  - `run_one_shot_algs.py`
  - `algs/fisher_avg.py`
  - `train_model.py`
  - `utils/compress_fisher.py`

## Data contract

- Data are generated from a sparse Gaussian binary model.
- Default dimensionality is `d=100`.
- The first `10` coordinates are signal and the remaining `90` coordinates are
  pure Gaussian noise.
- Default `signal_strength=0.7` and `noise_std=1.0`.
- Balanced-test Bayes accuracy is about `75.8%`.
- The default has `5` clients.
- The default non-IID split uses the original class-wise Dirichlet allocation
  with `alpha=0.1`.
- Keep `dim=100` and `signal_dim=10` unless the user explicitly asks to sweep
  data dimensionality.

## Original-pipeline contract

The parent repository provides:

- `data.py`: `SyntheticBinary`.
- `models.py`: `SyntheticMLP` and `SyntheticMLPDeep`.
- `main.py`: synthetic parameters and output naming.
- `utils/prediction_intervention.py`: trained-model feature intervention.
- `utils/feature_importance.py`: legacy supervised signal-recovery utilities.
- `scripts/summarize_synthetic_original.py` and plotting helpers.
- Preserved Slurm exports for small, 1,000-seed, and alpha-sweep runs.

Keep `LocalUpdate`, `run_one_shot_algs.py`, and `algs/fisher_avg.py` as close to
the authors' implementation as possible.

Current vector-input models are:

```text
SyntheticMLP:
100 -> 64 -> 32 -> 2
8610 parameters, bias=True

SyntheticMLPDeep:
100 -> 256 -> 128 -> 64 -> 32 -> 2
69154 parameters, bias=True
```

The original-pipeline methods are `fedavg`, `fedfisher_diag`, and
`fedfisher_kfac`.

## Analysis contracts

Preferred prediction-intervention analysis modifies one held-out input
coordinate at a time and scores changes in the trained model's own predictions.
Do not use true-label loss as its main score. Describe this path as model-based
feature intervention or prediction intervention.

The optional legacy feature-importance path ranks input dimensions and checks
whether the known signal set `{0, ..., synthetic_signal_dim - 1}` is recovered.
Describe it as supervised feature selection or signal recovery, not as a causal
treatment-effect or heterogeneous-treatment-effect experiment.

## Legacy and export surfaces

These files remain for audit or execution on a separately reviewed real Slurm
cluster; do not submit them from the current server:

- `scripts/run_synthetic_original_fedfisher.slurm`
- `scripts/run_synthetic_original_fedfisher_1000seeds.slurm`
- `scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm`

Any future execution path requires a reviewed fixed task name, strict parameter
schema, foreground adapter, resource profile, fixtures, bounded pilot, scientific
completion checks, and central registration before a formal request is created.

## Git hygiene and style

- Do not commit generated logs, `__pycache__`, large output CSVs, or figures
  unless the user explicitly asks.
- Prefer small source, test, configuration, and documentation changes.
- Keep defaults scientifically meaningful and expose explicit parameters for
  reviewed sweeps.
- Avoid hidden global state except deterministic random seeds.
- Avoid new heavyweight dependencies beyond PyTorch, NNGeometry, and the Python
  standard library without a separately reviewed environment change.
