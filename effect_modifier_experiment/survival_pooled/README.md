# Pooled survival probability benchmark

This additive benchmark tests whether predicted survival probabilities agree
with simulation truth **before** evaluating treatment contrasts. It compares
the accepted survival-v1 federated models with centralized models using the
same training records and independent test records. It does not change any
survival-v1 method, result, acceptance proof or producer provenance.

## Scientific design

The original simulation is retained: ten hospitals, 500–1,000 patients per
hospital, 100 Gaussian covariates, Weibull shape 1.5 and cumulative-hazard coefficient 0.12,
independent exponential censoring with rate 0.05, and evaluation time tau=3.
X1 and X2 are both prognostic/confounding variables and the only true treatment
effect modifiers. All covariates, including X2, are continuous in this project.
The two scenarios are IID and the existing hospital mean/scale covariate shift.
They share the same independent 10,000-person standard Gaussian test sample
within each seed. No new data-generating process or evaluation population is
introduced. See `../survival/data.py` for the exact propensity and hazard formulas.

For treatment a, the simulated truth is S_a(tau|X)=exp(-exp(eta_a(X))), where
eta_a is log cumulative hazard. Thus the link is log(-log(S)); equivalently the
event probability F=1-S has the complementary log-log link. The response is an
unclipped KM infinitesimal-jackknife pseudo-value of **survival** at tau, not a
binary event indicator or a log-transformed pseudo-value. The implementation
uses the existing working squared-error loss, not a Bernoulli likelihood.
There is no doubly robust augmentation, propensity weighting or AIPW estimator.

Each family has two centralized controls:

| Scope | Response construction | Interpretation |
|---|---|---|
| `site_local_pv` | Compute KM/IJ pseudo-values within each hospital, then concatenate training rows/labels | Compare aggregation/fitting while retaining the original labels |
| `pooled_km_pv` | Concatenate observed records, then compute one pooled KM/IJ response | Ordinary pooled survival analysis; also changes the labels |

Both scopes exclude test rows and simulation truth from training. Linear CV
recomputes responses separately in each training partition and each disjoint
validation partition. Full-training pseudo-values are not sliced into folds.
No patient-level pseudo-value is truncated to [0,1].

The linear control retains the accepted dsLasso r3 recipe: standardized
[A,X,A*X] inputs, interaction-only L1 penalty, five site-stratified folds,
30 penalty fractions from 1 to 0.001, minimum CV loss, full-data refitting,
16,000-iteration numerical safety bound and maximum backtracking step 64.
Every accepted endpoint must pass the saved first-order checks. Because the
distributed dsLasso objective is already a pooled sum of site losses, its
`site_local_pv` centralized control is an implementation-equivalence benchmark,
not an independent new statistical method. Arithmetic grouping by hospital
retains floating-point summation order; all data reside centrally.

The neural controls use the original float64 MLPs: shallow [64,32] or deep
[256,128,64,32], raw [A,X,A*X] inputs, the same seed-specific public initialization,
30 epochs, batch size 64 and SGD (learning rate 0.01, momentum 0.9). They do not
compute Fisher matrices, send uploads or aggregate models. Thirty pooled epochs
make 30 passes through all training rows, with approximately ten times as many
updates as one hospital's 30 local epochs. Epochs, processed rows and optimizer
updates are recorded separately; this is not a matched per-hospital update
budget or a proof of convergence.

Accepted historical dsLasso, FedFisher Diag and FedFisher KFAC models are restored
under their actual original producer commits. They are not retrained. Their
acceptance proof, manifests, artifacts, numerical semantics, central producing
attempts and original scientific source hashes are checked. Regenerated
training/test/truth fingerprints must match those historical artifacts before
any pooled fitting. Checkpoint replay and scientific evaluation occur inside
the registered task; later reporting reads persisted predictions only.

## Endpoints

Primary endpoints are RMSE, MAE and mean prediction bias against true S0 and S1,
evaluated separately on every common test row. The combined endpoint weights
the two arms equally. Factual accuracy uses each test patient's simulated
observed treatment and is reported separately. Ten fixed probability bins
record predicted and oracle mean survival for calibration diagnostics.

Paired FL–pooled RMS probability gaps measure how closely each FL prediction
matches each centralized control. A small gap does not itself prove agreement
with truth: both models may be inaccurate. The secondary contrast endpoint is
RMSE of eta1-eta0 on the log cumulative-hazard scale. This benchmark does not
rerun modifier-selection permutations. Low prediction accuracy is a finding,
not an automatic hidden failure threshold.

## Modules and immutable evidence

| Module | Responsibility |
|---|---|
| `protocol.py` / `requests.py` | Fixed recipes, deterministic seed coverage and reviewed source bindings |
| `inputs.py` | Frozen historical acceptance index and selected model/producer checks |
| `methods.py` | Centralized fitting and saved numerical validation |
| `evaluation.py` | Prediction followed by pure saved-array metrics |
| `computation.py` | Observed-data fitting and paired probability evaluation |
| `artifacts.py` / `runner.py` | Immutable manifests, completion markers and foreground lifecycle |
| `acceptance.py` | Saved-only pilot/formal verification and same-source gate |

Source files belong in the canonical repository. Completed models, prediction
arrays and results use `/data/del6500/FedFisher/effect_modifier_experiment/outputs/`
with independent `survival_pooled_v1_*` run IDs. Acceptance proofs and figures
use `/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-pooled-v1/`.
Requests and inventories belong below the FedFisher scratch scheduler outbox
and reports directories. Each seed's `model.npz` stores both pooled models,
all predicted eta/S arrays, factual treatment and the evaluator-only oracle
arrays. `result.json` contains arithmetic metrics, labels/model hashes, exact
historical input receipts, fitting budgets, measured stage clocks and RSS.
Invalid or partial outputs fail closed without overwrite.

## Pilot and formal execution

Pilot tasks cover three families, two scenarios and seeds 0 and 49: six jobs,
12 scientific units and 24 centralized fits. The formal plan covers seeds
0–299: 1,800 units and 3,600 centralized fits. Six family/scenario cohorts use
16 deterministic shards each by default (96 jobs); eight shards are also
schema-valid. Sharding changes no scientific seed or configuration.

The two additive task names are `effect-modifier-survival-pooled-pilot` and
`effect-modifier-survival-pooled-shard`. Their proposal is disabled until the
ServerScheduler operator reviews schema, CPU profile, fixtures and actual pilot
costs. Project publication creates an operator-review outbox, not a submitted
job. Keep canonical source frozen after publication through acceptance.

From the canonical root, using the existing science interpreter:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/prepare_survival_pooled_requests.py --cohort pilot
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/validate_survival_pooled.py --cohort pilot --inventory PILOT_INVENTORY
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/prepare_survival_pooled_requests.py --cohort formal --pilot-acceptance CANONICAL_PILOT_PROOF
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/validate_survival_pooled.py --cohort formal --inventory FORMAL_INVENTORY
```

Formal publication and execution both reverify the complete same-source pilot
proof. Neither a saved boolean nor a successful process exit opens the gate.
Admission and submission remain central operator actions. These centralized
comparators are simulation validation, not a privacy-preserving deployment or
an end-to-end one-round communication claim.

After scientific acceptance, the deterministic reporting command reads saved
prediction arrays and produces RMSE tables/figures and oracle calibration
curves. It does not load fitted models, run inference or regenerate data:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/make_survival_pooled_results.py --acceptance ACCEPTANCE_PROOF --output NEW_REPORT_DIRECTORY
```
