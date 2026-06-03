# Agent Guide

This folder contains synthetic binary-classification experiments for probing
FedFisher behavior on a controlled sparse Gaussian task.

There are two related but distinct experiment paths:

- Standalone path: files under `synthetic_binary_experiment/`.
- Original-pipeline path: root-level FedFisher code plus helper scripts under
  `scripts/`.

Keep these paths conceptually separate when editing or interpreting results.

## Scope

- Keep standalone synthetic implementation code inside
  `synthetic_binary_experiment/`.
- Generated standalone outputs should go under
  `synthetic_binary_experiment/outputs/`.
- Original-pipeline synthetic outputs should go under
  `synthetic_binary_experiment/outputs/original_fedfisher/`.
- Original-pipeline synthetic logs should go under
  `synthetic_binary_experiment/logs/original_fedfisher/`.
- Do not modify original FedFisher algorithm code unless the user explicitly
  asks. In particular, avoid changing:
  - `run_one_shot_algs.py`
  - `algs/fisher_avg.py`
  - `train_model.py`
  - `utils/compress_fisher.py`

The current original-pipeline adaptation only adds synthetic data/model entry
points and helper scripts. It does not change FedFisher aggregation logic.

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

## Standalone Experiment Contract

Entry point:

```bash
synthetic_binary_experiment/run_experiment.py
```

Main standalone methods:

- `pool`: centralized oracle baseline.
- `fedavg_oneshot`: one communication round with parameter averaging.
- `fedfisher_diag`: diagonal Fisher aggregation with FedAvg-initialized
  server-side Adam and validation iterate selection.
- `fedfisher_full`: LR-only complete Fisher matrix aggregation.
- `fedfisher_kfac`: MLP-only Kronecker-factored Fisher aggregation.
- Optional `fedavg_round_R`: multi-round FedAvg communication curve.

Correctness rules:

- All clients for a given seed must start from the exact same initial model
  state.
- FedAvg and FedFisher must use the same trained local models.
- Pooled training must start from the same initial state used by the clients.
- Pooled training is an oracle reference with separate centralized
  hyperparameters and validation-selected epoch count.
- Compute Fisher after local training, on each client's local data.
- Default local training should match the reference optimizer scheme:
  `SGD(lr=0.01, momentum=0.9, weight_decay=0)` with 30 local epochs.
- Keep Fisher damping centered at FedAvg, so unidentifiable coordinates do not
  get pulled toward zero by numerical regularization.
- Keep full Fisher LR-only unless the model is explicitly small enough.
- Keep K-FAC layer-wise; do not materialize a full MLP Fisher matrix.
- Report paired seed results whenever comparing FedFisher to FedAvg.

Standalone verification:

```bash
./.conda/bin/python synthetic_binary_experiment/tests/smoke_test.py
```

Manual quick run:

```bash
./.conda/bin/python synthetic_binary_experiment/run_experiment.py \
  --output-dir /tmp/fedfisher_synth_quick \
  --num-train 500 \
  --num-test 500 \
  --local-epochs 2 \
  --seeds 0 \
  --model-types lr \
  --splits iid
./.conda/bin/python synthetic_binary_experiment/summarize_results.py \
  --input /tmp/fedfisher_synth_quick/results.csv \
  --output /tmp/fedfisher_synth_quick/summary.csv
```

## Original-Pipeline Synthetic Contract

The original-pipeline synthetic path is used to test the synthetic task through
the original FedFisher implementation.

Parent-repository additions:

- `data.py`: adds `SyntheticBinary`.
- `models.py`: adds `SyntheticMLP` and `SyntheticMLPDeep`.
- `main.py`: adds synthetic CLI parameters and output file naming.
- `scripts/run_synthetic_original_fedfisher.slurm`: GPU array runner.
- `scripts/summarize_synthetic_original.py`: result summarizer.
- `scripts/plot_synthetic_original.py`: visualization script.

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

Slurm runner:

```bash
sbatch scripts/run_synthetic_original_fedfisher.slurm
```

The array is `0-19`:

```text
2 models x 2 splits x 5 seeds
```

Summarize and plot:

```bash
./.conda/bin/python scripts/summarize_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output synthetic_binary_experiment/outputs/original_fedfisher/summary.csv

./.conda/bin/python scripts/plot_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output-dir synthetic_binary_experiment/outputs/original_fedfisher/figures
```

Before submitting Slurm jobs, check:

```bash
bash -n scripts/run_synthetic_original_fedfisher.slurm
sbatch --test-only scripts/run_synthetic_original_fedfisher.slurm
./.conda/bin/python -c "import torch, torchvision, nngeometry; print(torch.__version__, torchvision.__version__)"
```

The Slurm script itself checks CUDA availability after allocation.

## GitHub Hygiene

- Do not commit generated logs, `__pycache__`, large output CSVs, or SVG figures
  unless the user explicitly wants results included in the repository.
- Prefer committing source scripts and documentation:
  - `synthetic_binary_experiment/*.py`
  - `synthetic_binary_experiment/synthetic_fedfisher/*.py`
  - `synthetic_binary_experiment/README.md`
  - `synthetic_binary_experiment/AGENTS.md`
  - relevant `scripts/*.py` and `scripts/*.slurm`
- Keep root algorithm files unchanged unless there is a clear, requested reason
  to modify the original FedFisher implementation.

## Style

- Prefer small helper functions over monolithic experiment scripts.
- Keep defaults scientifically meaningful, but expose CLI flags for sweeps.
- Avoid hidden global state except explicit random seeds.
- Avoid heavyweight dependencies beyond PyTorch, torchvision/nngeometry already
  used by the original repo, and the Python standard library.
