# Agent Guide

This folder now documents only the synthetic binary-classification experiment
that runs through the original FedFisher pipeline. The earlier standalone
PyTorch reimplementation has been removed.

## Scope

- Keep the synthetic task on the original FedFisher code path.
- Generated synthetic original-pipeline outputs should go under
  `synthetic_binary_experiment/outputs/original_fedfisher*`.
- Generated prediction-intervention outputs should go under
  `synthetic_binary_experiment/outputs/prediction_intervention*`.
- Generated synthetic original-pipeline logs should go under
  `synthetic_binary_experiment/logs/original_fedfisher*`.
- Generated prediction-intervention logs should go under
  `synthetic_binary_experiment/logs/prediction_intervention*`.
- Do not modify original FedFisher algorithm code unless the user explicitly
  asks. In particular, avoid changing:
  - `run_one_shot_algs.py`
  - `algs/fisher_avg.py`
  - `train_model.py`
  - `utils/compress_fisher.py`

The synthetic adaptation adds data/model entry points and helper scripts. It
does not change FedFisher aggregation logic.

## Data Contract

- Data are generated from a sparse Gaussian binary model.
- Default dimensionality is `d=100`.
- Default signal dimensions are the first `10` coordinates.
- The remaining `90` coordinates are pure Gaussian noise.
- Default `signal_strength=0.7`, `noise_std=1.0`.
- Balanced-test Bayes accuracy is about `75.8%`.
- Default clients are `5`.
- Default `noniid` split uses the original FedFisher class-wise Dirichlet
  allocation pattern with `alpha=0.1`.
- Keep `dim=100` and `signal_dim=10` unless the user explicitly asks to sweep
  data dimensionality.

## Original-Pipeline Synthetic Contract

Parent-repository additions:

- `data.py`: adds `SyntheticBinary`.
- `models.py`: adds `SyntheticMLP` and `SyntheticMLPDeep`.
- `main.py`: adds synthetic CLI parameters and output file naming.
- `scripts/run_synthetic_original_fedfisher.slurm`: small GPU array runner.
- `scripts/run_synthetic_original_fedfisher_1000seeds.slurm`: 1000-seed
  IID/non-IID runner.
- `scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm`:
  1000-seed alpha sweep runner.
- `scripts/summarize_synthetic_original.py`: result summarizer.
- `scripts/plot_synthetic_original.py`: IID/non-IID visualization script.
- `scripts/plot_synthetic_alpha_sweep.py`: alpha sweep visualization script.
- `utils/feature_importance.py`: legacy optional supervised signal-dimension
  recovery utilities for trained global models.
- `utils/prediction_intervention.py`: model-output feature intervention
  utilities for trained prediction models.

Do not change original FedFisher update logic when working on this path. The
goal is to keep `LocalUpdate`, `run_one_shot_algs.py`, and `algs/fisher_avg.py`
as close to the authors' implementation as possible.

Current vector-input synthetic models:

```text
SyntheticMLP:
100 -> 64 -> 32 -> 2
8610 parameters, bias=True

SyntheticMLPDeep:
100 -> 256 -> 128 -> 64 -> 32 -> 2
69154 parameters, bias=True
```

Current original-pipeline methods:

```text
fedavg
fedfisher_diag
fedfisher_kfac
```

Preferred prediction-intervention analysis:

```bash
./.conda/bin/python main.py \
  --dataset SyntheticBinary \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --prediction_intervention \
  --output_dir synthetic_binary_experiment/outputs/prediction_intervention
```

This analysis modifies one input coordinate at a time on the held-out test set
and scores features by changes in the trained model's own predictions. Do not
use true-label loss as the main score for this path. Keep the language framed
as model-based feature intervention or prediction intervention.

Legacy supervised feature-importance analysis:

```bash
./.conda/bin/python main.py \
  --dataset SyntheticBinary \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --feature_importance \
  --output_dir synthetic_binary_experiment/outputs/feature_importance
```

This analysis ranks input dimensions and checks whether the known signal set
`{0, ..., synthetic_signal_dim - 1}` is recovered. Keep the language framed as
supervised feature selection / signal recovery. Do not describe it as a causal
treatment-effect or HTE experiment.

Small Slurm runner:

```bash
sbatch scripts/run_synthetic_original_fedfisher.slurm
```

1000-seed runners:

```bash
sbatch scripts/run_synthetic_original_fedfisher_1000seeds.slurm
sbatch scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
```

Summarize and plot:

```bash
./.conda/bin/python scripts/summarize_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output synthetic_binary_experiment/outputs/original_fedfisher/summary.csv

./.conda/bin/python scripts/plot_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output-dir synthetic_binary_experiment/outputs/original_fedfisher/figures

./.conda/bin/python scripts/plot_synthetic_alpha_sweep.py
```

Before submitting Slurm jobs, check:

```bash
bash -n scripts/run_synthetic_original_fedfisher.slurm
bash -n scripts/run_synthetic_original_fedfisher_1000seeds.slurm
bash -n scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
sbatch --test-only scripts/run_synthetic_original_fedfisher.slurm
./.conda/bin/python -c "import torch, nngeometry; print(torch.__version__)"
```

The Slurm scripts themselves check CUDA availability after allocation.

## GitHub Hygiene

- Do not commit generated logs, `__pycache__`, large output CSVs, or SVG figures
  unless the user explicitly wants results included in the repository.
- Prefer committing source scripts and documentation:
  - `synthetic_binary_experiment/README.md`
  - `synthetic_binary_experiment/AGENTS.md`
  - relevant `scripts/*.py` and `scripts/*.slurm`
  - parent synthetic data/model entry points in `data.py`, `models.py`, and
    `main.py`
- Keep root algorithm files unchanged unless there is a clear, requested reason
  to modify the original FedFisher implementation.

## Style

- Keep defaults scientifically meaningful, but expose CLI flags for sweeps.
- Avoid hidden global state except explicit random seeds.
- Avoid heavyweight dependencies beyond PyTorch, nngeometry already used by the
  original repo, and the Python standard library.
