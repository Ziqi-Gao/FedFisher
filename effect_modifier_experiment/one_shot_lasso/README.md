# One-shot LASSO effect-modifier benchmarks

This directory documents two additive one-round baselines.  Neither replaces
the existing multi-round `dslassocov_*` implementation or its durable results.

## Families

### `one_shot_local_logistic_lasso_fedavg`

Each site independently fits a Bernoulli logistic LASSO.  For
`covariate_dim = d`, the intercept, treatment indicator, and `d` raw covariate
main effects are unpenalized; L1 is applied only to the `d` explicit
treatment-by-covariate interactions.  Standardization, five-fold CV, lambda
selection, and final refit occur inside the site.  If an accelerated path reaches its iteration limit with the legacy update-gate
flag false while both final prox-gradient and KKT residuals already pass the
unchanged threshold, the site performs a deterministic momentum-reset polish
for at most 50 iterations at the same lambda and tolerance.  Pre/post residuals
and coefficient drift are audited; a genuinely nonstationary fit is rejected.
The site back-transforms and
uploads one payload: int64 `site_idx`, int64 `n_k`, and `2 + 2*d` float64
raw-scale coefficients (202 coefficients at the formal default `d = 100`).
The server computes the strict sample-count weighted arithmetic mean.

The existing non-IID shard contract can create single-outcome sites, for which
the hard-label likelihood has no finite optimum when the adjustment block is
unpenalized.  Locally, site k therefore uses epsilon_k = 1/(n_k + 1) soft
binary targets, y_star = (1 - epsilon_k)y + epsilon_k/2, for fitting and CV.
This vanishing separation correction adds no coefficient penalty to the
intercept, A, or raw-X block and is recorded in every site audit and manifest.

### `one_shot_mlp_group_lasso_fedavg` (explicit interaction-input regularization)

Each site starts from the common `SyntheticMLP` or `SyntheticMLPDeep`
initialization and retains the existing 30 epochs, batch size 64, SGD learning
rate 0.01, and momentum 0.9.  After each SGD minibatch update, group soft
thresholding is applied only to the 100 first-layer columns corresponding to
`A*X1..A*X100`.  Each column across all first hidden units is one L2 group; all
other parameters are unpenalized.

The site-local penalty scale is the largest interaction-group cross-entropy
gradient norm at the common initialization.  Candidate ratios are
`0,1,3,10,30,100,300`.  A deterministic observed-class-stratified 80/20 split (with a class singleton
kept in training) selects the
largest lambda within one standard error of minimum validation cross-entropy,
after which the site refits from the common initialization on all local rows.
The site uploads one complete model vector and its sample count.  The server
uses the unchanged `run_one_shot_algs.get_fedavg_model` arithmetic average.

This is explicit interaction-input group regularization, not identifiable
effect-modifier selection: the unpenalized `A` and `X` inputs and nonlinear
hidden layers can still represent treatment-covariate interactions after an
explicit `A*Xj` column group is zero.  An independent matched zero-penalty CPU
FedAvg reference uses the same common state, explicit site DataLoader seed, and
full-site refit seed.  Regression tests require exact parameter equality between
that reference and the group path at lambda zero.  Neither path claims bitwise
equivalence to the legacy GPU/global-RNG FedAvg trajectory.

## Communication boundary

- Training communication is exactly one logical broadcast/upload round.
- Each site uploads once; byte accounting includes int64 `site_idx` and int64
  `n_k`.  Gradients, Hessians, Fisher information, training
  rows, and site callbacks are forbidden payload fields.
- The aggregation API accepts immutable payloads only.  Training-site objects
  are sealed after upload and may not be accessed again.
- This repository executes a single-process cross-silo simulation; it is not a
  deployment privacy or disclosure-control implementation.
- Global evaluation uses the coordinator-held common test set and records zero
  evaluation rounds.  Site-local intervention records its separate model and
  scalar-score exchange as one evaluation round and never reopens training
  data.

## Isolation and paths

RUN_IDs are namespaced by family and variant:

```
one_shot_local_logistic_lasso_fedavg_<variant>_...
one_shot_mlp_group_lasso_fedavg_<variant>_...
```

Durable output is restricted to
`/data/del6500/FedFisher/effect_modifier_experiment/outputs/<RUN_ID>/`.
The failed legacy formal RUN_IDs remain preserved without overwrite.  Repaired
formal cohorts end in `_stationarity_polish_300seeds`.
A formal multi-seed RUN_ID passes --seed-start 0 --seed-end 299 on every
cell invocation; the current cell seed must lie inside that manifest-bound range.
Every run has a self-consistent format-v2 manifest binding source provenance,
the common DGP/evaluation configuration, the family method configuration, and
the variant contract.  Existing files are never overwritten.

The formal CPU launcher is
`scripts/launch_effect_modifier_one_shot_lasso_cpu.sh`; its read-only status
command is `scripts/status_effect_modifier_one_shot_lasso_cpu.sh`.  It starts
four short-name Screen sessions, one per factorial variant, and each session
uses exactly one CPU thread while processing both isolated method families.
The total configured compute concurrency is therefore four threads.  No GPU
environment, GPU lock, or GPU scheduler is used.

Formal worker locks are isolated under
`/scr/del6500/FedFisher/effect_modifier_experiment/one_shot_cpu_locks/`.
Validated cells resume by skipping; partial cells, changed source/config,
invalid manifests, and hash mismatches fail closed without overwrite.  The
launcher and status command recognize only their four `ff_osp_*300` sessions
and never wipe or control foreign processes.  Full 300-seed launch still
requires no-cache syntax, all regression tests, and a same-source four-cell
seed-0 CPU smoke.
