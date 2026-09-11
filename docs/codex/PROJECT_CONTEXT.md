# FedFisher Project Context

This document is stable orientation for Codex sessions working in the canonical
FedFisher repository. It is not a live experiment-status record. Query Git,
scientific manifests, and the read-only ServerScheduler state before making any
claim about current jobs or result completeness.

`AGENTS.md` files are normative; this document is descriptive routing and
architecture reference. If they disagree, stop and follow the applicable
`AGENTS.md`.

## Canonical roots and authority

- Canonical source: `/home/del6500/projects/FedFisher`.
- Durable scientific data and artifacts: `/data/del6500/FedFisher`.
- Regenerable caches, runtime files, logs, and request outbox:
  `/scr/del6500/FedFisher`.
- `results/server-scheduler-migration/` is a preserved historical clone. Never
  edit, execute, import, register, or use it as a Codex working directory.
- Repository `results/compare/` and `results/table2/` contain small historical
  comparison material, not authoritative durable experiment output.

The root `AGENTS.md` is mandatory. Read the nearest nested `AGENTS.md` whenever
work touches an experiment subtree, even when the session started at the Git
root.

## Scientific scope

FedFisher is a binary-tabular federated-learning research repository centered
on one-shot aggregation, with an additional iterative dsLassoCov-style
benchmark. It contains these separate data paths:

1. `SyntheticBinary`: sparse Gaussian binary classification used for prediction
   and signal-recovery experiments.
2. `SyntheticEffectModifier`: the primary factorial experiment for recovering
   treatment effect modifiers under observed confounding, prognostic main
   effects, unequal site sizes, and noise covariates.
3. `LocalBinaryCSV`: a compatibility loader for local CSV data. It has no
   registered central task.
4. `SyntheticEffectModifierSCMV2`: an independent observed-data SCM under
   `effect_modifier_experiment/one_shot_effects/data.py`. It generates site,
   covariates, treatment and outcome in that order, with shared probability-scale
   CATE and a separate evaluator truth object. It does not change the original
   generator or its outcome-label partitioning.

The original Fisher-aggregation path implements FedAvg, diagonal FedFisher, and
KFAC FedFisher through `main.py`, `train_model.py`, `run_one_shot_algs.py`,
`algs/fisher_avg.py`, and `utils/compress_fisher.py`. Preserve this scientific
path separately from new adapters. The original two CPU task contracts are
LASSO benchmarks that reuse the effect-modifier data-generating process and
evaluation protocol; they do not run `fedfisher_diag` or `fedfisher_kfac`.
Current admission must be verified against the live central registration.

## Data-generating processes

### SyntheticBinary

The default has 100 features, 10 signal dimensions, five clients, signal
strength 0.7, and Gaussian noise standard deviation 1.0. IID data are generated
in balanced site partitions. Non-IID data use the original class-wise Dirichlet
partitioning. Evaluation uses an independent balanced test set.

### SyntheticEffectModifier

The default has 10 sites, independently sampled site sizes from 500 through
1000, and 100 raw Gaussian covariates. Model input is:

```text
[A, X1, ..., X100, A*X1, ..., A*X100]
```

Treatment assignment depends on `X1` and `X2` with propensity clipping. The
outcome has prognostic main effects for `X1` and `X2` and treatment interactions
`A*X1` and `A*X2`, with no treatment main effect. Only `X1` and `X2` are true
effect modifiers; the remaining raw covariates and interactions are noise.
IID partitioning preserves sampled site capacities. Non-IID partitioning is a
capacity-constrained, outcome-label Dirichlet allocation that also preserves
those capacities. The independent common test set defaults to 10,000 rows.

## Factorial variants

The four variants form one 2-by-2 scientific contract:

| Variant | Training covariate proposal | Intervention scope |
| --- | --- | --- |
| `control` | Common standard Gaussian | Global test set |
| `site_gaussian_shift` | Fixed site-specific Gaussian | Global test set |
| `local_permutation` | Common standard Gaussian | Disjoint site-local test shards |
| `site_gaussian_local_permutation` | Fixed site-specific Gaussian | Disjoint site-local test shards |

The Gaussian factor changes only the training covariate proposal and preserves
site capacities, outcome-label counts, and the common test set. The local factor
changes only intervention scope; its disjoint test shards are exhaustive and
their scalar feature scores are aggregated by sample size. Paired invariants
must prove that control/local and Gaussian/combined share the required data,
test, fit, payload, and aggregate-model fingerprints.

## Method families

- Original FedAvg: sample-size weighted model-vector averaging.
- Original FedFisher diagonal and KFAC: Fisher-based one-shot aggregation on
  the preserved original implementation path.
- dsLassoCov-style CPU benchmark: a Python cross-silo simulation of federated
  logistic LASSO, not official R/DataSHIELD dsLassoCov and not a deployment
  privacy boundary. It penalizes only treatment-interaction coefficients and
  uses federated standardization, site-stratified cross-validation, FISTA,
  backtracking, and stationarity checks.
- One-shot local logistic LASSO + FedAvg: each site fits and rescales its own
  penalized logistic model, uploads one coefficient vector plus site weight,
  and the server performs one sample-size weighted aggregation.
- One-shot MLP group-LASSO + FedAvg: local `SyntheticMLP` and
  `SyntheticMLPDeep` models penalize first-layer groups attached to explicit
  interaction inputs, upload one model vector, and use the original FedAvg
  aggregation. This is explicit interaction-input regularization, not a claim
  of identifiable effect-modifier selection through all nonlinear paths.

## Evaluation contract

Primary recovery is based on raw-covariate treatment-contrast intervention,
not absolute interaction coefficients. Counterfactual model inputs must remain
internally consistent:

```text
A=1: [1, X, X]
A=0: [0, X, 0]
tau_hat(X) = treatment score - control score
```

Each raw `Xi` is permuted and the change in `tau_hat` is ranked. The primary
metric is `permute/tau_intervention_logit`, summarized by Top-2 recovery and
AUROC across 100 raw covariates. Probability-scale treatment contrast and
direct intervention on explicit `A*X` columns are auxiliary diagnostics.

## Experiment units and artifacts

The dsLassoCov experiment has 300 seeds, two splits per seed, and four variants.
The one-shot LASSO experiment has six cells per seed and variant: linear,
`SyntheticMLP`, and `SyntheticMLPDeep`, each under IID and non-IID partitioning.

Each scheduler-era namespace has a format-v2 `run_manifest.env`. Provenance
binds Git commit, dirty state, Git-status hash, the tracked plus non-ignored
untracked source-tree hash, and canonical run/method/variant configuration
hashes. Result creation is fail-closed and rejects symlinks, partial output,
invalid identity, and overwrite.

A process exit code is never sufficient scientific completion evidence. A unit
may be skipped only after validating its run manifest, current source
provenance, semantic result identity, artifact hashes, and task-specific
completion marker. Final four-variant comparisons also require paired
invariants.

## Current source routing

- Data generation and partitioning: `data.py`.
- Models: `models.py`.
- Original FedFisher/FedAvg: `train_model.py`, `run_one_shot_algs.py`, `algs/`.
- Common effect-modifier evaluation: `utils/prediction_intervention.py`.
- dsLassoCov implementation: `effect_modifier_experiment/dslassocov_*.py` and
  `manage_dslassocov_cpu_run.py`.
- One-shot LASSO implementation:
  `effect_modifier_experiment/one_shot_*.py` and
  `manage_one_shot_lasso_cpu_run.py`.
- Scientific contracts: `effect_modifier_experiment/*_run_contract.py`.
- Central foreground adapter:
  `scripts/fedfisher_server_scheduler_adapter.py`.
- Parameter schemas and request examples:
  `effect_modifier_experiment/server_scheduler/`.
- Tests and scheduler fixtures: `tests/` and
  `tests/fixtures/server_scheduler/`.
- Additive multiround FedAvg, homogeneous R methods, common probability-scale
  comparison, timing, and immutable comparison cells:
  `effect_modifier_experiment/causal_comparison/`. Read its `AGENTS.md` and
  `README.md` for scientific recipes and scope; task registration remains a
  separate operator-controlled gate. Its probability-scale recovery handles
  cutoff ties explicitly and does not replace historical logit endpoints.
- FedFisher-EM, DR-Sieve and their independent common-protocol baselines:
  `effect_modifier_experiment/one_shot_effects/`. Its `AGENTS.md`, `README.md`
  and `SPECIFICATION.md` describe the observed-data interfaces, T/V/H split,
  original-site target weights, serialized one-upload boundary, SCMV2 and legacy
  compatibility, and separately defined absolute and squared scores. Modules
  separate data, basis/nuisance fitting, Fisher aggregation/compression,
  evaluation, reporting, immutable artifacts and scheduler composition.
  Its `ose_v1_*_pilot` namespaces never reuse historical completion evidence.
  Stage expansion requires source-matched full-gradient diagnostic evidence;
  task admission and live status remain operator-owned, recorded in the handoff.

Launchers, local workers, Screen sessions, CPU/GPU locks, direct Python compute
commands, and Slurm files are preserved legacy or export surfaces. They are not
authorized paths for new computation. Preserve scientific workers, manifests,
results, completion markers, and legacy audit material unless the user gives a
separate, exact deletion instruction.

## Target architecture, not current layout

Future refactoring should separate deterministic data generation, models,
method implementations, evaluation, experiment protocols, artifact lifecycle,
and scheduler orchestration. The intended dependency direction is:

```text
orchestration -> experiments -> data + methods + evaluation -> artifacts
```

Methods must not generate data, choose scheduler resources, or own filesystem
namespaces. Data generators must not call methods. Experiment protocols compose
data, methods, and evaluation. This is a target design; the canonical repository
is still the current mostly flat module layout until a separately verified
migration is committed.
