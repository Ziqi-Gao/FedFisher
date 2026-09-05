# dsLassoCov-style CPU benchmark

This directory documents an additive Python benchmark for the existing
`SyntheticEffectModifier` factorial experiment. The method name is:

> **dsLassoCov-style federated logistic LASSO (Python benchmark; not official
> R/DataSHIELD dsLassoCov).**

It simulates cross-silo site aggregation in one Python process. Individual
training rows remain in site objects during fitting and the optimization
coordinator combines only site-level loss and gradient contributions, but this
simulation does **not** provide a deployed DataSHIELD disclosure-control or
privacy boundary. The runner computes dataset fingerprints for reproducibility,
which is another reason not to interpret it as a privacy implementation.

## Statistical design

The benchmark directly calls the existing `data.get_dataset` implementation
with `SyntheticEffectModifier`; it does not reproduce or approximate the DGP.
The defaults therefore remain 10 sites, independent site sizes from
`500..1000`, 100 raw covariates, the existing IID/non-IID partitions, the
fixed site-Gaussian proposal, one common test set, and the existing site-local
test shards.

For outcome `Y`, the fitted score is

```text
logit P(Y=1 | A,X) =
    intercept + beta_A A + sum_j gamma_j X_j + sum_j theta_j A*X_j
```

`intercept`, `A`, and all 100 raw-X main effects are unpenalized. Only the 100
interaction coefficients are subject to `lambda * ||theta||_1`. At every
proximal-gradient iteration, each site computes its own Bernoulli logistic
loss and gradient; the coordinator aggregates their sums by the total sample
count. Training rows are never concatenated by this optimizer.

All non-intercept columns are standardized from training-only site
count/sum/sum-of-squares summaries. CV uses deterministic, outcome-stratified
folds made independently inside every site. Validation logistic loss is
aggregated by validation sample count. The minimum-loss lambda is selected;
ties within `1e-12` choose the largest lambda. The final fit records the full
lambda path, per-fold/per-lambda convergence, standardization and
back-transformation, RNG seeds, proximal-gradient mapping, and explicit KKT
residuals. A result cannot receive a done marker unless every selected CV fit
and the selected full-data fit pass the configured stationarity threshold.
Test data are not accepted by the fitting API and are used only after lambda
selection.

## Evaluation

The fitted original-scale linear score is wrapped as a two-logit PyTorch model
and passed to the unchanged treatment-contrast intervention functions in
`utils/prediction_intervention.py`.

- Primary: permute each of 100 raw covariates, rebuild internally consistent
  `[1,X,X]` and `[0,X,0]` inputs, and report Top-2 and AUROC for
  `tau_intervention_logit`.
- Factorial local-permutation cells use the unchanged disjoint site test shards
  and sample-size-weighted scalar aggregation.
- Auxiliary only: absolute fitted `A*Xj` coefficients are reported in a
  separately labelled JSON diagnostic.

Control/local-permutation must have identical training and fit fingerprints.
Site-Gaussian/combined must also have identical training and fit fingerprints.
All four cells must have the same common-test fingerprint for a seed/split.

## Difference from the published method

The published dsLassoCov method jointly fits penalized features and
unpenalized covariates with proximal-gradient updates formed from weighted
site losses/gradients, and the official implementation is in the R
`dsMTLClient`/`dsMTLBase` DataSHIELD stack. This benchmark borrows that joint
penalized/unpenalized and site-gradient structure, but deliberately differs:

- it uses a Bernoulli logistic objective specialized to this binary-outcome
  effect-modifier DGP;
- it defines the penalized candidates as `A*X1..A*X100` and treats
  intercept/A/raw-X as the adjustment block;
- it is a single-process Python cross-silo simulation without DataSHIELD
  server-side disclosure controls;
- its primary endpoint is the repository's raw-X tau-intervention Top-2/AUROC,
  not the paper's coefficient-selection endpoint;
- the default path has 30 lambdas rather than the 100-lambda real-data example
  reported in the paper.

Source: [Scientific Reports article and method equations](https://www.nature.com/articles/s41598-026-48845-0).

## CPU-only execution and storage

All new runs use the registered ServerScheduler task
`effect-modifier-dslassocov-shard`. One central job owns one deterministic
shard, stays in the foreground, receives one physical CPU core and fixed 4096
MiB host memory. Parameters are limited to the four reviewed
variants, `shard_count` in `1,2,4`, and a zero-based `shard_index` below that
count. Run IDs, paths, seed range `0..299`, splits, and scientific
hyperparameters are derived by the adapter and cannot be supplied by a caller.

ServerScheduler owns CPU selection, affinity, admission, logs, and global
concurrency. The adapter sets every BLAS/OpenMP pool to one thread and calls the
existing manager and runner directly. It skips only a fully validated seed
marker and refuses partial or invalid artifacts. Scheduler-era output from the
current common-source cohort uses the fixed
`dslassocov_<variant>_server_scheduler_v2r2_300seeds` namespace. The original
`server_scheduler_v2_300seeds` and later `server_scheduler_v2r1_300seeds`
namespaces are retained as historical audit evidence and must not be resumed,
overwritten, or combined with v2r2 results.

`scripts/prepare_fedfisher_scheduler_requests.py --phase pilot` creates one
bounded 0/4 shard per variant from a clean Git tree. Only after operator
submission, central termination, current-source scientific validation, and
paired-invariant checks may the helper create the other three shards per
variant with `--phase remaining --confirm-pilots-validated`. Outbox files are
not submitted jobs.

The shell launcher, Screen workers, and lock paths described below are retained
only to audit pre-cutover runs. They are not valid submission paths for new
work.

The CPU launcher is separate from the existing GPU launcher/slot/worker chain.
It does not query GPUs, set a device visibility variable, use GPU locks, or
control any foreign process. Defaults are two CPU workers with one BLAS/Torch
thread each; both values are manifest-hashed, each is capped at four, and their
product is hard-limited to four total compute threads.

Durable artifacts:

```text
/data/del6500/FedFisher/effect_modifier_experiment/outputs/<RUN_ID>/
```

Logs, runtime files, and caches:

```text
/scr/del6500/FedFisher/effect_modifier_experiment/logs/<RUN_ID>/
/scr/del6500/FedFisher/effect_modifier_experiment/runtime/<RUN_ID>/
```

Nonblocking launch and per-seed locks live only under
`/scr/del6500/FedFisher/effect_modifier_experiment/cpu_locks/<RUN_ID>/`.
The seed lock covers both splits and the validated done marker and is released
on skip, success, failure, or signal. Screen workers cannot inherit the launch
lock descriptor. These locks are separate from, and never touch, the existing
GPU scheduler locks. The CPU scripts control only their exact RUN_ID sessions
and never stop, reprioritize, or otherwise alter another experiment.

Every output directory starts with a format-v2 manifest containing commit,
dirty/status/source hashes (including untracked source), common run config,
benchmark/CV/optimizer config, and explicit variant config. Existing nonempty
directories without a matching manifest are rejected. Partial split artifacts
are never overwritten or counted as complete.

## Legacy full 300-seed commands (do not launch)

```bash
EFFECT_MODIFIER_VARIANT=control \
  RUN_ID=dslassocov_control_300seeds \
  bash scripts/launch_effect_modifier_dslassocov_cpu.sh
EFFECT_MODIFIER_VARIANT=site_gaussian_shift \
  RUN_ID=dslassocov_site_gaussian_shift_300seeds \
  bash scripts/launch_effect_modifier_dslassocov_cpu.sh
EFFECT_MODIFIER_VARIANT=local_permutation \
  RUN_ID=dslassocov_local_permutation_300seeds \
  bash scripts/launch_effect_modifier_dslassocov_cpu.sh
EFFECT_MODIFIER_VARIANT=site_gaussian_local_permutation \
  RUN_ID=dslassocov_site_gaussian_local_permutation_300seeds \
  bash scripts/launch_effect_modifier_dslassocov_cpu.sh
```

Status:

```bash
EFFECT_MODIFIER_VARIANT=<variant> RUN_ID=<run-id> \
  bash scripts/status_effect_modifier_dslassocov_cpu.sh
```

After all four are complete:

```bash
python -B scripts/check_effect_modifier_dslassocov_invariants.py
```

## Legacy bounded smoke template (do not launch directly)

The smoke retains 100 raw covariates, 10 sites, and site sizes `500..1000`,
but bounds test size, folds, lambda count, and intervention repeats:

```bash
EFFECT_MODIFIER_VARIANT=site_gaussian_local_permutation \
RUN_ID=dslassocov_site_gaussian_local_permutation_smoke_<unique> \
SEED_START=0 SEED_END=0 CPU_WORKERS=1 THREADS_PER_WORKER=1 \
SYNTHETIC_NUM_TEST=500 CV_FOLDS=2 LAMBDA_COUNT=4 \
PREDICTION_INTERVENTION_REPEATS=1 RUN_IN_FOREGROUND=1 \
  bash scripts/launch_effect_modifier_dslassocov_cpu.sh
```
