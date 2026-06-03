# Synthetic Binary FedFisher Experiments

This directory contains synthetic binary-classification experiments for
understanding when one-shot FedFisher helps or hurts relative to one-shot
FedAvg.

There are two experiment paths:

1. `synthetic_binary_experiment/` standalone implementation.
   This is a self-contained PyTorch implementation used for controlled
   debugging, CPU runs, pooled-oracle baselines, full Fisher for logistic
   regression, and detailed diagnostics.
2. Original FedFisher pipeline adaptation.
   This reuses the repository's original `main.py`, `LocalUpdate`,
   `run_one_shot_algs.py`, `algs/fisher_avg.py`, and `nngeometry` Fisher path.
   Only synthetic data/model entry points are added to the parent pipeline.

The second path is the best choice when the goal is to compare against the
original author's FedFisher implementation as closely as possible.

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
- `mild`: client positive-label priors are `[0.30, 0.40, 0.50, 0.60, 0.70]`.
- `noniid`: generate a balanced global training set, then split examples with
  the class-wise Dirichlet allocation used by the original FedFisher data code.
  The default concentration is `alpha = 0.1`.

The test set is always balanced with positive-label prior `0.5`.

## Project Structure

```text
synthetic_binary_experiment/
  README.md
  AGENTS.md
  run_experiment.py              # standalone synthetic experiment entry point
  run_main_cpu.slurm             # standalone CPU batch run
  summarize_results.py           # standalone result summarizer
  plot_results.py                # standalone result visualizer
  tests/smoke_test.py            # standalone quick regression test
  synthetic_fedfisher/
    data.py                      # standalone synthetic data generation
    models.py                    # standalone LR/MLP models
    training.py                  # standalone training and Fisher estimators
    federated.py                 # standalone FedAvg/FedFisher implementations
  outputs/
    main_cpu/                    # completed standalone CPU outputs
    original_fedfisher/          # completed original-pipeline synthetic outputs
  logs/
    original_fedfisher/          # original-pipeline Slurm logs
```

Related parent-repository files for the original-pipeline adaptation:

```text
data.py                          # adds SyntheticBinary dataset branch
models.py                        # adds SyntheticMLP and SyntheticMLPDeep
main.py                          # accepts SyntheticBinary CLI parameters
scripts/run_synthetic_original_fedfisher.slurm
scripts/summarize_synthetic_original.py
scripts/plot_synthetic_original.py
```

The original FedFisher aggregation code remains in:

```text
run_one_shot_algs.py
algs/fisher_avg.py
train_model.py
utils/compress_fisher.py
```

These algorithm files are not changed by the synthetic adaptation.

## Standalone Synthetic Experiment

This path is useful for debugging because it includes pooled oracle training,
full Fisher for logistic regression, explicit communication counts, validation
selection diagnostics, and optional multi-round FedAvg.

### Smoke Test

From the repository root:

```bash
./.conda/bin/python synthetic_binary_experiment/tests/smoke_test.py
```

### Main CPU Run

```bash
sbatch synthetic_binary_experiment/run_main_cpu.slurm
```

This writes:

```text
synthetic_binary_experiment/outputs/main_cpu/results.csv
synthetic_binary_experiment/outputs/main_cpu/summary.csv
synthetic_binary_experiment/outputs/main_cpu/figures/
```

### Manual Standalone Run

```bash
./.conda/bin/python synthetic_binary_experiment/run_experiment.py \
  --output-dir synthetic_binary_experiment/outputs/default \
  --num-train 10000 \
  --num-test 10000 \
  --dim 100 \
  --signal-dim 10 \
  --signal-strength 0.7 \
  --noise-std 1.0 \
  --num-clients 5 \
  --dirichlet-alpha 0.1 \
  --local-epochs 30 \
  --optimizer sgd \
  --weight-decay 0.0 \
  --pool-epochs 80 \
  --pool-lr 1e-3 \
  --pool-weight-decay 1e-2 \
  --pool-optimizer adam \
  --pool-val-fraction 0.2 \
  --pool-patience 8 \
  --fisher-damping 1e-6 \
  --fisher-server-steps 2000 \
  --fisher-server-lr 1e-2 \
  --fisher-server-eval-every 100 \
  --fisher-val-size 500 \
  --seeds 0 1 2 3 4 \
  --model-types lr mlp \
  --splits iid noniid
```

Summarize and plot:

```bash
./.conda/bin/python synthetic_binary_experiment/summarize_results.py \
  --input synthetic_binary_experiment/outputs/default/results.csv \
  --output synthetic_binary_experiment/outputs/default/summary.csv

./.conda/bin/python synthetic_binary_experiment/plot_results.py \
  --input synthetic_binary_experiment/outputs/default/results.csv \
  --output-dir synthetic_binary_experiment/outputs/default/figures
```

## Original FedFisher Pipeline Synthetic Experiment

This path runs synthetic data through the original repository training and
aggregation pipeline. It is designed to answer whether surprising synthetic
results come from the standalone reimplementation or from FedFisher behavior
itself.

### Models

The original paper experiments use convolutional local models (`LeNet`, `CNN`,
and `ResNet18`). The synthetic tabular data need vector-input models, so the
parent `models.py` adds:

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

### Methods

Each original-pipeline synthetic task runs:

```text
fedavg
fedfisher_diag
fedfisher_kfac
```

The FedFisher update logic is the original code path in `run_one_shot_algs.py`
and `algs/fisher_avg.py`.

### Slurm Run

```bash
sbatch scripts/run_synthetic_original_fedfisher.slurm
```

The array has 20 tasks:

```text
2 models x 2 splits x 5 seeds
```

Task layout:

```text
0-4    SyntheticMLP,     iid,    seeds 0-4
5-9    SyntheticMLP,     noniid, seeds 0-4
10-14  SyntheticMLPDeep, iid,    seeds 0-4
15-19  SyntheticMLPDeep, noniid, seeds 0-4
```

Outputs:

```text
synthetic_binary_experiment/logs/original_fedfisher/
synthetic_binary_experiment/outputs/original_fedfisher/
```

The Slurm script includes a preflight import/CUDA check before training:

```text
import torch, torchvision, nngeometry
assert torch.cuda.is_available()
```

### Summarize and Visualize

```bash
./.conda/bin/python scripts/summarize_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output synthetic_binary_experiment/outputs/original_fedfisher/summary.csv

./.conda/bin/python scripts/plot_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output-dir synthetic_binary_experiment/outputs/original_fedfisher/figures
```

Generated files:

```text
synthetic_binary_experiment/outputs/original_fedfisher/summary.csv
synthetic_binary_experiment/outputs/original_fedfisher/figures/original_compare_summary.csv
synthetic_binary_experiment/outputs/original_fedfisher/figures/original_accuracy_bars.svg
synthetic_binary_experiment/outputs/original_fedfisher/figures/original_gain_heatmap.svg
synthetic_binary_experiment/outputs/original_fedfisher/figures/original_seed_pairs.svg
```

## Result Columns

Standalone `results.csv` columns include:

- `accuracy`: test accuracy.
- `loss`: test cross-entropy.
- `bayes_accuracy`: theoretical Bayes accuracy for the balanced test set.
- `gap_to_pool`: method accuracy minus pooled oracle accuracy for the same
  seed/model/split.
- `gain_over_fedavg`: method accuracy minus one-shot FedAvg accuracy for the
  same seed/model/split.
- `uplink_scalars`: approximate client-to-server scalar communication.
- `fisher_selected_step`: validation-selected FedFisher server step.

Original-pipeline summaries use accuracy percentages:

- `accuracy_mean_pct`
- `accuracy_std_pct`
- `gain_over_fedavg_mean_pct`
- `gain_over_fedavg_std_pct`
- `seed_wins`

## Interpretation Notes

- The synthetic data are linearly generated, so deeper networks are not needed
  for Bayes optimality. The deep MLP experiment is intended to probe
  overparameterized local training and one-shot aggregation, not representation
  limits of the data.
- IID settings are expected to be less favorable for FedFisher because local
  optima are already aligned enough for FedAvg.
- Non-IID settings stress one-shot FedAvg through client drift; these are the
  settings where Fisher-weighted aggregation is most informative.
