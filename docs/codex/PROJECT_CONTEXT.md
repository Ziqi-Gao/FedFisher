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

FedFisher studies federated effect-modifier discovery for binary and survival
outcomes. The paper objective combines efficient discovery under separate
hospital records and one-round FedFisher fitting with substantive findings in
real EHR data. The simulation code supports that objective; it does not establish
a completed EHR study or an end-to-end one-round EHR workflow. Original binary
prediction and supervised signal-recovery experiments are also preserved.
The separate data paths are:

1. `SyntheticBinary`: sparse Gaussian binary classification used for prediction
   and signal-recovery experiments.
2. `SyntheticEffectModifier`: the primary factorial experiment for recovering
   treatment effect modifiers under observed confounding, prognostic main
   effects, unequal site sizes, and noise covariates.
3. `LocalBinaryCSV`: a compatibility loader for numeric binary-outcome CSV data.
4. `SyntheticEffectModifierSCMV2`: an independent observed-data SCM under
   `effect_modifier_experiment/one_shot_effects/data.py`. It generates site,
   covariates, treatment and outcome in that order, with shared probability-scale
   CATE and a separate evaluator truth object. It does not change the original
   generator or its outcome-label partitioning.
5. `SyntheticEffectModifierSurvivalV1`: the independent survival generator in
   `effect_modifier_experiment/survival/data.py`. Site-first IID/covariate-shift
   scenarios retain X1/X2 confounding and effect modification, use Weibull PH
   event times with independent censoring, and keep evaluator truth separate.
   Its site-local IJ pseudo-values and log cumulative-hazard contrast do not
   reuse binary outcome-label allocation or binary result namespaces.

The original Fisher-aggregation path implements FedAvg, diagonal FedFisher, and
KFAC FedFisher through `main.py`, `train_model.py`, `run_one_shot_algs.py`,
`algs/fisher_avg.py`, and `utils/compress_fisher.py`. Preserve this scientific
path separately from new adapters. The original two CPU task contracts are
LASSO benchmarks that reuse the effect-modifier data-generating process and
evaluation protocol; they do not run `fedfisher_diag` or `fedfisher_kfac`.
Current admission must be verified against the live central registration.

## MLP architecture for new experiments

Use exactly one ReLU hidden layer with **8 neurons** for every newly designed
MLP experiment. The existing binary model name is `SyntheticMLPSmall`; preserve
the outcome-specific output head. For survival with `[A,X,A*X]` and 100 raw
covariates, the intended network is **201 -> 8 -> 1**, where the scalar output
is log cumulative hazard and the survival mean remains `exp(-exp(eta))`.

Do not schedule additional shallow `[64,32]` or deep `[256,128,64,32]` MLP
experiments. Their named models, fixed historical recipes, checkpoints and
results remain available for historical interpretation. Do not redefine those
names as 8-neuron models or count their acceptance evidence as an MLP8 pilot.
Linear comparators remain separate from this neural architecture choice.

Architecture changes require an independent recipe, scientific namespace and
bounded pilot under the usual scheduler gates. In particular, a new survival
FL-versus-pooled comparison must fit fresh MLP8 federated and pooled models;
the historical shallow/deep replay benchmark does not implement that comparison.
Keep data generation, pseudo-value construction, link, endpoints and training
settings unchanged unless the user separately requests a change.

The complete survival FL-versus-pooled comparison checks survival-probability
accuracy first, then log cumulative-hazard treatment contrasts and raw-X
permutation modifier discovery. All four MLP8 models must report Top-2 recovery
of both X1/X2 and AUROC using the original five-repeat, tie-aware survival
evaluator. Prediction-only completion evidence cannot authorize a discovery
cohort. Each recipe keeps separate task names, namespaces and pilot proofs.

## Original binary data-generating processes

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

## Original binary factorial variants

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
- Survival adapters: nonlinear dsLasso-style interaction LASSO and scalar-output
  Diag/KFAC FedFisher use site-local survival pseudo-values, squared-error loss
  and `exp(-exp(eta))` as the survival mean. Their Gaussian working Fisher is
  separate from categorical likelihood information and from the sampling
  covariance of dependent pseudo-values. The survival README defines the
  fitting communication boundary and independent simulation evaluation.

## Original binary evaluation contract

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

| Area | Source and scientific guide |
| --- | --- |
| Original binary/base | Root `data.py`, `models.py`, `main.py`, `train_model.py`, `run_one_shot_algs.py`, `algs/` and `utils/`; [guide](../../synthetic_binary_experiment/README.md) |
| Original binary effect modifiers | `effect_modifier_experiment/dslassocov_*.py`, `one_shot_*.py`, their managers and `*_run_contract.py`; [guide](../../effect_modifier_experiment/README.md) |
| Additional method comparison | [causal_comparison](../../effect_modifier_experiment/causal_comparison/README.md): multiround FedAvg, homogeneous R methods, common probability-scale evaluation and timing |
| Additional one-shot methods | [one_shot_effects](../../effect_modifier_experiment/one_shot_effects/README.md): FedFisher-EM, DR-Sieve, T/V/H interfaces, SCMV2 and legacy compatibility |
| Saved-model diagnostics | [one_shot_effects/diagnostics](../../effect_modifier_experiment/one_shot_effects/diagnostics/README.md): gradient audits, approximation gates and source-bound replay |
| Original Fisher CPU timing | [fedfisher_timing](../../effect_modifier_experiment/fedfisher_timing/README.md): isolated Diag/KFAC timing, immutable artifacts and pilot acceptance |
| Survival | [survival](../../effect_modifier_experiment/survival/README.md): generator, IJ responses, links, methods, evaluation, protocol, artifacts and acceptance |
| Survival FL-versus-pooled comparison | [survival_mlp8](../../effect_modifier_experiment/survival_mlp8/README.md): fresh single-layer 8-neuron Diag/KFAC and paired pooled controls; probability checks followed by original Top-2/AUROC discovery, separate recipe/proof gates; historical [survival_pooled](../../effect_modifier_experiment/survival_pooled/README.md) remains separate |
| Shared infrastructure | [infrastructure](../../effect_modifier_experiment/infrastructure/README.md): standard-library artifact, provenance, request-publication and read-only execution-evidence helpers |
| Shared scientific mechanisms | [components](../../effect_modifier_experiment/components/README.md): timing ledgers, array/model fingerprints and existing tie-aware recovery arithmetic; experiment scales remain local |
| Scheduling implementation | [scheduling](../../effect_modifier_experiment/scheduling/README.md): catalog, strict validation, runtime setup, lazy dispatch and publication composition; deterministic request plans belong to experiment packages |
| Reporting | [reporting](../../reporting/README.md): reusable presentation modules behind compatible script commands |
| Scheduler boundary | [adapter](../../scripts/fedfisher_server_scheduler_adapter.py) and [schemas/proposals/examples](../../effect_modifier_experiment/server_scheduler/) |
| Validation | [tests](../../tests/) and [scheduler fixtures](../../tests/fixtures/server_scheduler/) |

Read the nearest `AGENTS.md` and scientific guide before changing an experiment.
Each package owns its estimand, method configuration, semantic validator and
result namespace. Shared helpers do not make probability, logit and survival
contrasts interchangeable, or authorize reuse of another package's completion
evidence. Scientific and central admission gates remain separate.

Launchers, local workers, Screen sessions, CPU/GPU locks, direct Python compute
commands, and Slurm files are preserved legacy or export surfaces. They are not
authorized paths for new computation. Preserve scientific workers, manifests,
results, completion markers, and legacy audit material unless the user gives a
separate, exact deletion instruction.

## Shared infrastructure and remaining legacy layout

`effect_modifier_experiment/infrastructure/` extracts neutral operations:
`artifacts.py` handles paths, strict JSON and immutable byte publication;
`provenance.py` handles source/configuration hashes and environment manifests;
`requests.py` publishes immutable outbox bundles with the inventory last;
`evidence.py` reads central producer/retry/audit evidence without scheduler
API calls. This layer uses the Python standard library and imports no scientific
experiment packages. Survival and timing reuse these helpers; compatibility
exports preserve existing helper import paths.

The extracted dependency direction is:

```text
scripts / scheduler adapter -> experiment protocol + runner + acceptance
experiment runner           -> experiment data + methods + evaluation
experiment lifecycle code   -> shared infrastructure -> Python standard library
```

Methods must not generate data, choose scheduler resources, or own filesystem
namespaces. Data generators must not call methods. Experiment protocols compose
data, methods and evaluation. Scientific result validation remains with each
experiment; infrastructure does not define a universal configuration or method
API. The original root modules and historical scientific cross-imports remain;
this extraction does not claim a complete reorganization of the flat pipeline.

## Modular source composition

Comparison and standalone timing expose path-free `computation.py` modules.
Model/signal recipes reuse those modules rather than another experiment's runner.
One-shot effects exposes `data_composition.py` and `prerequisites.py` so diagnostics
and validators can reuse data/evidence composition without importing a runner.
Compatibility exports retain existing public paths.

Experiment `requests.py` and `recovery_requests.py` own deterministic scientific
request plans. `scheduling/publication/` composes source review, experiment
admission and immutable outbox publication. Acceptance readers use experiment
plans instead of script wrappers. `components/` owns shared observations,
fingerprints and ranking arithmetic; it does not select a contrast scale or
replace an experiment's scientific contract.

See [architecture](../architecture.md) and [development](../development.md) for
the complete dependency direction and validation workflow. These documents
describe stable structure; changing scientific and migration evidence belongs
only in the handoff.
