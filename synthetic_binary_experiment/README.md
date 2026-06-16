# Synthetic Binary FedFisher Pipeline

This directory documents the synthetic binary-classification experiment that
runs through the repository's original FedFisher training and aggregation
pipeline.

The previous standalone reimplementation has been removed. The remaining path
reuses the original repository code:

```text
main.py
data.py
models.py
run_one_shot_algs.py
algs/fisher_avg.py
train_model.py
utils/compress_fisher.py
```

Synthetic data/model entry points are added in the parent repository, while the
FedFisher aggregation logic stays on the original code path.

## Data Model

For label `y in {0, 1}`:

```text
sign = 2y - 1
x = sign * mu + epsilon
epsilon ~ N(0, noise_std^2 I)
```

Default data parameters:

```text
dim = 100
signal_dim = 10
mu_j = signal_strength / sqrt(signal_dim), j < signal_dim
mu_j = 0, j >= signal_dim
signal_strength = 0.7
noise_std = 1.0
num_train = 10000
num_test = 10000
num_clients = 5
```

Only the first 10 coordinates carry signal; the remaining 90 coordinates are
pure Gaussian noise. The balanced test distribution has theoretical Bayes
accuracy `Phi(signal_strength / noise_std)`, about `75.8%` with the defaults.

## Client Splits

- `iid`: every client has positive-label prior `0.5`.
- `noniid`: generate a balanced global training set, then split examples with
  the class-wise Dirichlet allocation used by the original FedFisher data code.
  The default concentration is `alpha = 0.1`.

The test set is always balanced with positive-label prior `0.5`.

## Project Structure

```text
synthetic_binary_experiment/
  README.md
  AGENTS.md
  logs/
    original_fedfisher/                       # original-pipeline Slurm logs
    original_fedfisher_1000seeds/
    original_fedfisher_alpha_sweep_1000seeds/
  outputs/
    original_fedfisher/                       # original-pipeline outputs
    original_fedfisher_1000seeds/
    original_fedfisher_alpha_sweep_1000seeds/
```

Related parent-repository files:

```text
data.py                                      # SyntheticBinary dataset branch
models.py                                    # SyntheticMLP, SyntheticMLPDeep
main.py                                      # SyntheticBinary CLI parameters
scripts/run_synthetic_original_fedfisher.slurm
scripts/run_synthetic_original_fedfisher_1000seeds.slurm
scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
scripts/summarize_synthetic_original.py
scripts/plot_synthetic_original.py
scripts/plot_synthetic_alpha_sweep.py
utils/feature_importance.py
```

## Models

Only vector-input synthetic models remain:

```text
SyntheticMLP:
100 -> 64 -> 32 -> 2
8610 parameters with bias=True

SyntheticMLPDeep:
100 -> 256 -> 128 -> 64 -> 32 -> 2
69154 parameters with bias=True
```

Both are ordinary `nn.Linear`/`ReLU` networks and are only used for
`SyntheticBinary`.

## Methods

Each original-pipeline synthetic task runs:

```text
fedavg
fedfisher_diag
fedfisher_kfac
```

The FedFisher update logic is the original code path in `run_one_shot_algs.py`
and `algs/fisher_avg.py`.

## Feature Importance / Signal Recovery

The parent `main.py` supports an optional `--feature_importance` analysis for
SyntheticBinary. This ranks input dimensions after each trained global model is
obtained and evaluates whether the top-ranked dimensions recover the known
signal coordinates. By construction, dimensions
`0, ..., synthetic_signal_dim - 1` are informative for the binary label, while
the remaining dimensions are Gaussian noise.

This is a supervised feature-selection / signal-dimension-recovery experiment,
not a causal treatment-effect or HTE experiment. The main score is
permutation-ablation loss increase on the held-out test set. Auxiliary scores
include zero ablation, first-layer weight norm, aggregated-local-Fisher-weighted
first-layer importance, and final-global-model Fisher-weighted first-layer
importance.

Feature-importance outputs are additional CSVs only and do not change the
standard one-shot result CSV:

```text
<run_prefix>_feature_importance.csv
<run_prefix>_feature_importance_summary.csv
```

A centralized `pooled` baseline is included in these feature-importance outputs
by default. It trains the same model on the full global training set and gives a
reference for how well the current model and scoring methods can recover the
known signal dimensions without federated splitting or one-shot aggregation.

## Slurm Runs

Small original-pipeline run:

```bash
sbatch scripts/run_synthetic_original_fedfisher.slurm
```

1000-seed IID/non-IID run:

```bash
sbatch scripts/run_synthetic_original_fedfisher_1000seeds.slurm
```

1000-seed alpha sweep:

```bash
sbatch scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
```

The Slurm scripts include preflight import/CUDA checks before training:

```text
import torch, nngeometry
assert torch.cuda.is_available()
```

## Summarize And Visualize

Small run:

```bash
./.conda/bin/python scripts/summarize_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output synthetic_binary_experiment/outputs/original_fedfisher/summary.csv

./.conda/bin/python scripts/plot_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output-dir synthetic_binary_experiment/outputs/original_fedfisher/figures
```

Alpha sweep:

```bash
./.conda/bin/python scripts/plot_synthetic_alpha_sweep.py
```

Original-pipeline summaries use accuracy percentages:

- `accuracy_mean_pct`
- `accuracy_std_pct`
- `gain_over_fedavg_mean_pct`
- `gain_over_fedavg_std_pct`
- `seed_wins`
- `win_rate`

## Interpretation Notes

- The synthetic data are linearly generated, so deeper networks are not needed
  for Bayes optimality. The deep MLP experiment probes overparameterized local
  training and one-shot aggregation, not representation limits of the data.
- IID settings are expected to be less favorable for FedFisher because local
  optima are already aligned enough for FedAvg.
- Non-IID settings stress one-shot FedAvg through client drift; these are the
  settings where Fisher-weighted aggregation is most informative.
