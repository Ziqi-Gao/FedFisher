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
scripts/setup_lab_server_env.sh
scripts/launch_synthetic_effect_modifier_300seeds_local.sh
scripts/status_synthetic_effect_modifier_local.sh
scripts/run_synthetic_effect_modifier_1000seeds.slurm  # legacy Slurm export
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

## Execution and submission policy

The canonical code root is `/home/del6500/projects/FedFisher`. All new compute
work on the lab server must be submitted through ServerScheduler. Direct
Python, Slurm, GNU Screen, and local resource-lock launch commands shown later
in this README are retained only as legacy/export documentation and are not
authorized execution paths on the current host.

The stable project entrypoint is:

```text
/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py
```

The first migration stage registers only these CPU tasks:

- `effect-modifier-dslassocov-shard`: one physical CPU core, at least 4096 MiB
  host memory, `shard_count` in `1,2,4`.
- `effect-modifier-one-shot-lasso-shard`: one physical CPU core, at least 8192
  MiB host memory, `shard_count` in `8,16`.

Both accept one of `control`, `site_gaussian_shift`, `local_permutation`, or
`site_gaussian_local_permutation`, plus a valid zero-based `shard_index` below
`shard_count`. Run IDs, paths, seed ranges, models, splits, and scientific
hyperparameters are fixed by project code rather than request parameters.
Example requests and parameter schemas live under
`effect_modifier_experiment/server_scheduler/`.

GPU submission is disabled. Any other program is also blocked until it has a
reviewed task profile, strict parameter schema, foreground adapter path,
fixture tests, and an approved pilot. ServerScheduler owns CPU/GPU selection
and global concurrency. FedFisher owns scientific provenance, output
validation, completion markers, and at-least-once resume behavior.

All artifacts created before this canonical scheduler cutover are retained for
audit but are non-authoritative and must not be resumed or combined with new
results. See
`effect_modifier_experiment/server_scheduler/legacy-results-policy.md`.

## Install

On the current lab server, create the isolated Blackwell-compatible environment
under the protocol-approved scratch root:

```bash
bash scripts/setup_lab_server_env.sh
```

This installs PyTorch 2.7.1 with CUDA 12.8 because the server's RTX PRO 6000
Blackwell GPUs require `sm_120` kernels. The environment and package caches are
regenerable files under `/scr/del6500/FedFisher`; code stays in the repository
and durable experiment outputs stay under `/data/del6500/FedFisher`.

For another compatible host, `pip install -r requirements.txt` installs the
same pinned dependencies. The checked-in code does not include generated data,
outputs, environments, or logs.

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
- `--num_clients`: number of sites; defaults to `10` for
  `SyntheticEffectModifier` and `5` otherwise.
- `--num_rounds`: local-training and aggregation rounds, default `1`.
- `--local_epochs`: local client epochs, default `30`.
- `--synthetic_split`: `iid` or `noniid`, default `noniid`.
- `--synthetic_num_train`: SyntheticBinary training examples, default `10000`.
- `--synthetic_num_test`: held-out test examples, default `10000`.
- `--synthetic_dim`: SyntheticBinary feature dimension, default `100`.
- `--synthetic_signal_dim`: number of SyntheticBinary informative coordinates,
  default `10`.
- `--synthetic_signal_strength`: SyntheticBinary class-mean signal norm,
  default `0.7`.
- `--synthetic_noise_std`: SyntheticBinary Gaussian noise standard deviation,
  default `1.0`.
- `--effect_modifier_covariate_dim`: number of raw covariates, default `100`.
- `--effect_modifier_signal_dim`: number of true effect modifiers, default `2`.
- `--effect_modifier_signal_strength`: interaction signal norm, default `2.0`.
- `--effect_modifier_main_effect_strength`: prognostic main-effect norm,
  default `1.0`.
- `--effect_modifier_confounding_strength`: treatment-assignment coefficient
  norm, default `1.0`.
- `--effect_modifier_intercept`: binary-outcome logit intercept, default `0.0`.
- `--effect_modifier_treatment_prob`: treatment probability at `X1=X2=0`,
  default `0.5`.
- `--effect_modifier_propensity_clip`: lower/upper treatment-propensity overlap
  bound, default `0.05`.
- `--effect_modifier_site_min_samples` and
  `--effect_modifier_site_max_samples`: inclusive per-site training-size range,
  defaults `500` and `1000`.
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

`SyntheticEffectModifier` tests whether the trained models recover two true
treatment effect modifiers in the presence of observed confounding, first-order
outcome effects, unequal site sizes, and 98 noise covariates.

The generated input remains:

```text
[A, X1, ..., X100, A*X1, ..., A*X100]
```

Treatment assignment and outcome generation are:

```text
logit P(A = 1 | X) =
  logit(base_treatment_prob) + confounding_strength / sqrt(2) * (X1 + X2)

logit P(Y = 1 | A, X) =
  intercept
  + main_effect_strength / sqrt(2) * (X1 + X2)
  + signal_strength / sqrt(2) * (A*X1 - A*X2)
```

Treatment propensities are clipped to preserve overlap. `X1` and `X2` are the
only observed confounders, prognostic variables, and true effect modifiers;
`X3..X100` and their treatment interactions are noise. The default 10 site
sizes are independently sampled from the integers `500..1000`.

Primary recovery perturbs each raw `Xi`, rebuilds consistent counterfactual
inputs `[1, X, X]` and `[0, X, 0]`, and ranks changes in the model-estimated
treatment contrast. Evaluation reports Top-2 recovery and AUROC over all 100 raw
covariates. Direct interaction-column intervention remains secondary.

Smoke-test command:

```bash
python main.py \
  --dataset SyntheticEffectModifier \
  --model SyntheticMLP \
  --num_clients 10 \
  --synthetic_split noniid \
  --alpha 0.1 \
  --effect_modifier_covariate_dim 100 \
  --effect_modifier_signal_dim 2 \
  --effect_modifier_signal_strength 2.0 \
  --effect_modifier_main_effect_strength 1.0 \
  --effect_modifier_confounding_strength 1.0 \
  --effect_modifier_propensity_clip 0.05 \
  --effect_modifier_site_min_samples 500 \
  --effect_modifier_site_max_samples 1000 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --prediction_intervention \
  --prediction_intervention_modes permute zero \
  --prediction_intervention_repeats 5 \
  --output_dir /data/del6500/FedFisher/effect_modifier_experiment/outputs/prediction_intervention
```

Legacy lab-server runner (retained for audit; do not launch new work):

```bash
bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
bash scripts/status_synthetic_effect_modifier_local.sh
```

These commands describe the superseded pre-ServerScheduler workflow. They use
detached GNU Screen workers and project-local GPU admission, so they are not a
valid current-host submission path. See `effect_modifier_experiment/README.md`
for the scientific contract and current migration status.

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

## Legacy Quest Slurm Runs (export only; not a submission path)

The following runners belong to the original SyntheticBinary Quest workflow.
They are retained for reproducibility but are not valid submission commands on
`fsmzhaolabgpu1`, where Slurm is not installed. The current effect-modifier
experiment uses the local launcher documented above.

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

These legacy scripts are not normalized: some embed Quest account, partition,
and GPFS paths while others expect them at submission time. Review and adapt
them only when exporting an old SyntheticBinary workflow to a real Slurm
cluster; do not invoke them on the current lab server.

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
