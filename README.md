# FedFisher Binary Tabular Pipeline

This repository is adapted from the FedFisher codebase for one-shot federated
learning. The pipeline in this branch is scoped to binary tabular
classification, with a synthetic generator for sanity checks and a CSV loader
for local datasets.

The original image datasets and image models have been removed. The remaining
entry points are:

```text
data.py                                      # SyntheticBinary, SyntheticEffectModifier, and LocalBinaryCSV data loaders
models.py                                    # SyntheticMLP and SyntheticMLPDeep
main.py                                      # one-shot FL runner
utils/feature_importance.py                  # legacy supervised signal-dimension recovery utilities
utils/prediction_intervention.py             # model-output prediction intervention utilities
scripts/run_synthetic_original_fedfisher.slurm
scripts/run_synthetic_original_fedfisher_1000seeds.slurm
scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
scripts/run_synthetic_effect_modifier_1000seeds.slurm
scripts/summarize_synthetic_original.py
scripts/plot_synthetic_original.py
scripts/plot_synthetic_alpha_sweep.py
```

FedFisher aggregation remains on the original implementation path:

```text
run_one_shot_algs.py
algs/fisher_avg.py
utils/compress_fisher.py
```

## Install

Clone the repository and install the minimal dependencies in your own Python
environment:

```bash
git clone git@github.com:Ziqi-Gao/FedFisher.git
cd FedFisher
pip install -r requirements.txt
```

The checked-in code does not include local datasets, generated outputs, logs,
or cluster-specific paths.

## Main Arguments

Required arguments:

- `--dataset`: `SyntheticBinary`, `SyntheticEffectModifier`, or
  `LocalBinaryCSV`.
- `--model`: `SyntheticMLP` or `SyntheticMLPDeep`.
- `--algs_to_run`: one or more algorithms. The synthetic pipeline is intended
  for `fedavg`, `fedfisher_diag`, and `fedfisher_kfac`.

Useful optional arguments:

- `--seed`: random seed, default `0`.
- `--alpha`: Dirichlet concentration for non-IID splitting, default `0.1`.
- `--num_clients`: number of clients, default `5`.
- `--num_rounds`: local-training and aggregation rounds, default `1`.
- `--local_epochs`: local client epochs, default `30`.
- `--synthetic_split`: `iid` or `noniid`, default `noniid`.
- `--synthetic_num_train`: training examples, default `10000`.
- `--synthetic_num_test`: test examples, default `10000`.
- `--synthetic_dim`: feature dimension, default `100`.
- `--synthetic_signal_dim`: number of informative coordinates, default `10`.
- `--synthetic_signal_strength`: class-mean signal norm, default `0.7`.
- `--synthetic_noise_std`: Gaussian noise standard deviation, default `1.0`.
- `--effect_modifier_covariate_dim`: number of raw covariates for
  `SyntheticEffectModifier`, default `100`.
- `--effect_modifier_signal_dim`: number of true effect modifiers, default
  `10`.
- `--effect_modifier_signal_strength`: interaction signal norm, default `2.0`.
- `--effect_modifier_intercept`: binary-outcome logit intercept, default `0.0`.
- `--effect_modifier_treatment_prob`: treatment assignment probability,
  default `0.5`.
- `--feature_importance`: run supervised signal-dimension recovery for trained
  global models, default off.
- `--feature_importance_repeats`: permutation repeats per feature, default `5`.
- `--feature_importance_modes`: ablation modes, default `permute zero`.
- `--feature_importance_no_pooled_baseline`: skip the centralized pooled
  feature-importance baseline.
- `--prediction_intervention`: run model-based feature intervention on the
  held-out SyntheticBinary test set, default off.
- `--prediction_intervention_modes`: intervention modes, default
  `permute zero`.
- `--prediction_intervention_repeats`: permutation repeats per feature,
  default `5`.
- `--prediction_intervention_include_local_models`: also run the intervention
  analysis for each trained local client model.
- `--prediction_intervention_no_pooled_baseline`: skip the centralized pooled
  prediction-intervention baseline.

Synthetic sanity-check example:

```bash
python main.py \
  --dataset SyntheticBinary \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

## Prediction Intervention / Signal Recovery

The main signal-recovery analysis is a model-based feature intervention. The
pipeline first trains the FL/MLP prediction models. On the independent
SyntheticBinary test set, each input dimension is modified one at a time while
all other dimensions are fixed. The outcome is the trained model's own predicted
label, class-1 probability, and binary logit score. The feature effect is the
prediction change caused by that intervention.

This analysis does not use true labels to compute feature scores. Labels are
used only for reporting baseline test accuracy and for checking whether the
ranking recovers the known signal coordinates `{0, ..., synthetic_signal_dim -
1}`.

Metrics include:

- `abs_logit_change`: mean absolute change in `z1 - z0`.
- `signed_logit_change`: mean signed change in `z1 - z0`; ranked by absolute
  value.
- `abs_prob_change`: mean absolute change in class-1 probability.
- `signed_prob_change`: mean signed change in class-1 probability; ranked by
  absolute value.
- `flip_rate`: fraction of examples whose predicted label changes.
- `margin_drop`: drop in confidence for the model's original predicted class;
  ranked by absolute value.

Recommended primary rankings are `permute + margin_drop` and
`permute + abs_logit_change`. `flip_rate` is an intuitive auxiliary metric.

When `--prediction_intervention` is enabled, outputs are additional files only:

```text
<run_prefix>_prediction_intervention.csv
<run_prefix>_prediction_intervention_summary.csv
<run_prefix>_prediction_intervention_model_summary.csv
```

A centralized `pooled` baseline is included by default so the federated models
can be compared with a model trained directly on all training examples.

Smoke-test command:

```bash
python main.py \
  --dataset SyntheticBinary \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --synthetic_num_train 10000 \
  --synthetic_num_test 10000 \
  --synthetic_dim 100 \
  --synthetic_signal_dim 10 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --prediction_intervention \
  --prediction_intervention_modes permute zero \
  --prediction_intervention_repeats 5 \
  --output_dir synthetic_binary_experiment/outputs/prediction_intervention
```

## Effect-Modifier Interaction Recovery

`SyntheticEffectModifier` is an additive binary-outcome simulation for testing
whether the trained models recover treatment-by-covariate interaction signals.
It does not replace `SyntheticBinary`.

The generated input is:

```text
[A, X1, ..., Xp, A*X1, ..., A*Xp]
```

The outcome logit contains no raw `X` main effects and no treatment main effect:

```text
logit P(Y = 1 | A, X) = intercept + sum_{j in S} gamma_j * A * Xj
```

With defaults, `p=100` and `|S|=10`, so the interaction block is zero-indexed
`101..200` and the true effect-modifier interaction columns are `101..110`.
Recovery metrics rank only the interaction block `A*X1..A*Xp`; raw `X` columns
are not used to evaluate effect-modifier recovery.

Smoke-test command:

```bash
python main.py \
  --dataset SyntheticEffectModifier \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --effect_modifier_covariate_dim 100 \
  --effect_modifier_signal_dim 10 \
  --effect_modifier_signal_strength 2.0 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --prediction_intervention \
  --prediction_intervention_modes permute zero \
  --prediction_intervention_repeats 5 \
  --output_dir effect_modifier_experiment/outputs/prediction_intervention
```

Large-array runner:

```bash
sbatch scripts/run_synthetic_effect_modifier_1000seeds.slurm
```

## Legacy Supervised Feature Importance

The SyntheticBinary generator makes the first `synthetic_signal_dim` input
coordinates informative for the binary label and the remaining coordinates
Gaussian noise. The feature-importance experiment ranks input dimensions using
trained global MLPs and evaluates whether the ranking recovers the known signal
coordinates `{0, ..., synthetic_signal_dim - 1}`. This is a supervised feature
selection / signal recovery experiment, not a causal treatment-effect or HTE
experiment.

Methods include:

- `ablation_permute_loss`: primary score; permute one test-set input column and
  measure the cross-entropy loss increase.
- `ablation_zero_loss`: set one test-set input column to zero and measure loss
  increase.
- `weight_norm`: first-layer input-column weight norm.
- `fisher_weighted`: first-layer weight norm weighted by the aggregated local
  diagonal Fisher already computed during federated training.
- `global_fisher_weighted`: first-layer weight norm weighted by diagonal Fisher
  recomputed for the final global model on the global training set.

When `--feature_importance` is enabled, outputs are additional files only:

```text
<run_prefix>_feature_importance.csv
<run_prefix>_feature_importance_summary.csv
```

The detailed file contains one row per feature and method. The summary file
reports signal recovery metrics such as top-k hits, precision, signal/noise
ranks, and AUROC. A centralized `pooled` baseline is included by default in
these feature-importance outputs so the federated models can be compared with a
model trained directly on all training examples.

Smoke-test command:

```bash
python main.py \
  --dataset SyntheticBinary \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --synthetic_num_train 10000 \
  --synthetic_num_test 10000 \
  --synthetic_dim 100 \
  --synthetic_signal_dim 10 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --feature_importance \
  --feature_importance_repeats 5 \
  --output_dir synthetic_binary_experiment/outputs/feature_importance
```

## Using Local Data

Use `LocalBinaryCSV` when you clone the repository and want to run the pipeline
on your own local data. The loader expects numeric CSV files:

- labels must be encoded as `0` and `1`
- all non-label, non-client columns are used as model features
- train and test feature columns must match
- local data files can live anywhere; putting them under `data/` keeps them
  ignored by Git

If your training CSV already has a client column, pass it with
`--local_client_col`. The pipeline will use those groups as federated clients.
The test CSV should contain the same feature columns and label column, without
the client column.

```bash
python main.py \
  --dataset LocalBinaryCSV \
  --local_train_csv data/my_train.csv \
  --local_test_csv data/my_test.csv \
  --local_has_header \
  --local_label_col label \
  --local_client_col client_id \
  --model SyntheticMLP \
  --local_epochs 30 \
  --output_dir results/my_local_run \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

If your training CSV does not have client IDs, the pipeline will create clients
from the global training set. Use IID partitioning:

```bash
python main.py \
  --dataset LocalBinaryCSV \
  --local_train_csv data/my_train.csv \
  --local_test_csv data/my_test.csv \
  --local_has_header \
  --local_label_col label \
  --local_partition iid \
  --num_clients 5 \
  --model SyntheticMLP \
  --output_dir results/my_local_iid \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

Or use Dirichlet non-IID partitioning controlled by `--alpha`:

```bash
python main.py \
  --dataset LocalBinaryCSV \
  --local_train_csv data/my_train.csv \
  --local_test_csv data/my_test.csv \
  --local_has_header \
  --local_label_col label \
  --local_partition noniid \
  --alpha 0.1 \
  --num_clients 5 \
  --model SyntheticMLPDeep \
  --output_dir results/my_local_noniid \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

For CSVs without a header, omit `--local_has_header` and pass a zero-based
column index. The default label column is `-1`, meaning the last column.

## Slurm Runs

Small run:

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

The Slurm scripts intentionally omit account and partition directives. Pass
cluster-specific settings with `sbatch --account ... --partition ...` or your
site defaults.

## Results

Outputs are written under:

```text
synthetic_binary_experiment/outputs/original_fedfisher*
```

Logs are written under:

```text
synthetic_binary_experiment/logs/original_fedfisher*
```

Summarize and plot:

```bash
python scripts/summarize_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output synthetic_binary_experiment/outputs/original_fedfisher/summary.csv

python scripts/plot_synthetic_original.py
python scripts/plot_synthetic_alpha_sweep.py
```

## Notes

- Generated outputs, logs, caches, and `__pycache__` files should not be
  committed unless explicitly requested.
- Keep FedFisher implementation changes separate from dataset/model plumbing.
- `algs/fisher_avg.py` and `utils/compress_fisher.py` are the FedFisher
  implementation files to keep stable when packaging this synthetic pipeline.
