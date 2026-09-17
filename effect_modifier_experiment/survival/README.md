# Survival pseudo-observation experiment

[Project navigation](../../README.md#project-navigation) ·
[Experiment index](../README.md#experiment-navigation) ·
[Shared infrastructure](../infrastructure/README.md)

The paper aims to demonstrate efficient treatment-effect-modifier discovery
when hospitals cannot centralize patient records and FedFisher fitting uses one
communication round, and to report substantive, supported effect-modifier
findings in real EHR data. The real-data findings are a central contribution.
This experiment supplies the known-truth survival validation needed before that
application. It does not by itself establish clinical findings or superiority.

This is the independent `SyntheticEffectModifierSurvivalV1` experiment. It
preserves the original binary experiments, their model implementations, result
namespaces and completion evidence. Ava's recommendation to retain the original
confounder setup motivates the following defaults:

| Component | Definition |
| --- | --- |
| Covariates | 100 Gaussian variables; X1 and X2 are confounders, prognostic variables and the only true modifiers |
| Other variables | X3–X100 are noise in the primary simulation |
| Hospitals | 10 sites; independent integer sizes from 500 through 1000 |
| Treatment | Original X1/X2-dependent propensity, clipped to [.05,.95] |
| Outcome | Weibull proportional-hazards event time with independent exponential censoring |
| Baseline parameters | Weibull shape 1.5, lambda .12; censoring **rate** .05; evaluation time tau=3 |
| Primary effect scale | Log cumulative-hazard difference, equal to conditional log hazard ratio in this PH DGP |
| Test population | Independent 10,000 standard-Gaussian rows, shared across scenarios and methods |

There are 100 covariates and **two true confounders**, not 100 confounders.
The value tau=3 has simulation units; it does not prescribe a clinical horizon.
The observed censoring fraction is measured, not assumed to be five percent.

## Data and estimand

The original normalized coefficients and absence of a treatment main effect
are retained:

```
logit P(A=1 | X) = (X1 + X2) / sqrt(2)       [then propensity clipping]
r(A,X) = (X1 + X2) / sqrt(2) + A * 2*(X1-X2) / sqrt(2)
T = {E / [.12 * exp(r(A,X))]}^(1/1.5),      E ~ Exponential(rate=1)
C ~ Exponential(rate=.05)
observed_time = min(T,C); event = 1[T<=C]
S_a(tau | X) = exp(-exp(eta_a))
eta_a = log(.12 * tau^1.5) + r(a,X)
Delta(X) = eta_1 - eta_0 = 2*(X1-X2)/sqrt(2)
```

Sites are generated before treatment and outcomes. `iid` uses a common
standard Gaussian covariate distribution; `covariate_shift` uses the fixed
site mean-by-scale Gaussian design. Site sizes and independent test rows
match across these scenarios. The binary outcome-label Dirichlet allocation
is not applicable to this survival generator. These scenarios have new
identities and do not claim to reproduce the historical binary non-IID split
or the four-arm binary factorial experiment.

Observed site rows expose X, A, observed time and event only. Counterfactual
truth resides in a separate evaluator object and never enters model fitting.

## Pseudo-observations and link

Each hospital independently constructs its Kaplan–Meier curve and survival
pseudo-values at tau. `pseudo.py` implements the infinitesimal jackknife used
by R `survival::pseudo(..., type="surv")`:

```
PV_i = S_hat(tau) + n * dS_hat(tau)/d(weight_i), evaluated at all weights=1.
```

It is an infinitesimal-jackknife approximation, not an exact leave-one-out
jackknife. The implementation includes tied events/censoring, events exactly
at tau, R-style near-tie correction, and endpoint handling. Finite values
outside [0,1] are retained. There is no pooled KM preprocessing or exchange of
individual survival records.

The response mean is `mu=exp(-exp(eta))`. Ava's survival link is
`eta=log(-log(S))`, which is ordinary complementary-log-log applied to event
risk `F=1-S`. Applying an ordinary cloglog inverse directly to a survival
response reverses the meaning. Neither link is applied to individual PVs.

| Binary implementation | Survival adaptation |
| --- | --- |
| Binary response | Continuous survival PV, possibly outside [0,1] |
| Logistic/two-class output | Scalar eta and survival mean exp(-exp(eta)) |
| Bernoulli cross entropy | Half mean squared error, matching Ava's squared-error criterion up to scale |
| Categorical Fisher | Gaussian **working-model** expected Fisher / mean-Jacobian GGN |
| Logit treatment contrast | eta(A=1,X)-eta(A=0,X) on the log cumulative-hazard scale |

The censoring contract is independent censoring within each site, as in the
simulation. Delayed entry, case weights, repeated subjects, competing risks,
and informative-censoring corrections require additional designs. The real
EHR analysis must use its actual treatment definition, cohort, outcome and
clinical horizon; these are not inferred from the simulation defaults.

Primary implementation references:
[R pseudo documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/pseudo.html),
[pseudo.R](https://github.com/therneau/survival/blob/master/R/pseudo.R),
[residuals.survfit.R](https://github.com/therneau/survival/blob/master/R/residuals.survfit.R),
[aeqSurv.R](https://github.com/therneau/survival/blob/master/R/aeqSurv.R).
Analytic examples and independent numerical weighted-KM derivatives are the
Python fixtures. Executed R/Python parity is a separate validation check.

## Method contracts

`dslasso` is the Python dsLassoCov-style **survival adaptation**, not the official
R/DataSHIELD package. Its design is `[intercept,A,X,A*X]`; only A*X coefficients
receive L1 penalties. The server aggregates local counts, moments, losses and
gradients. This is an iterative communication baseline, not a one-shot method.

Five within-site, outcome-independent CV folds compare 30 lambda fractions
from 1 to .001. Each training fold computes its own local PVs, standardization
and lambda maximum. Validation folds compute separate local PVs, so their
survival records cannot enter training responses. Sample-weighted validation
MSE selects the fraction; exact ties prefer the stronger penalty. Full local
training data are then used to recompute PVs and refit. All tuning and refit
communication and time are counted. First-order KKT and proximal-gradient
checks are required. Squared error with this nonlinear mean is nonconvex:
stationarity is not evidence of a global optimum or good effect recovery.

`fedfisher_diag_{shallow,deep}` and `fedfisher_kfac_{shallow,deep}` use the
original hidden widths (64/32 and 256/128/64/32) and a scalar eta output.
Local fixed-budget SGD uses 30 epochs, batch size 64, learning rate .01 and
momentum .9. Sites share public initialization; its output bias uses the public
simulation baseline, never a pooled response mean. There is no test-driven
selection or outcome-dependent server broadcast during fitting.

For working variance sigma^2=1, the local curvature is the site mean of
`J_mu J_mu' / sigma^2`. Diag uses its exact diagonal; KFAC approximates each
layer block by the product of activation and mean-output-gradient factors,
including bias coordinates. This is a Gaussian working Fisher, **not** the
true likelihood information or sampling covariance of dependent KM PVs.

Each site uploads one serialized parameter/curvature/count summary. The server
receives bytes and public configuration only, and minimizes

```
sum_k (n_k/N)/2 * (w-w_k)' F_k (w-w_k)
    + rho/2 * ||w-w_public||^2,            rho=1e-6.
```

Diag has a closed-form solution; KFAC uses matrix-free conjugate gradients.
The explicit public-initialization ridge makes singular working curvature
solvable. Residual tolerance and the 2,000-iteration safety limit are distinct;
exhaustion is not convergence or scientific completion. This regression adapter
is separately identified from the original categorical, quantized,
validation-accuracy-selected aggregation path. No compression approximation is
added. Defaults are fixed reviewable recipes, not claimed optimal choices.

Ava's final adaptive LASSO on estimated individual contrasts is a different
selection stage. This experiment retains the project's raw-X intervention
evaluation and does not silently substitute her second-stage adaptive LASSO
for either the dsLasso penalty or FedFisher feature scoring.

## Evaluation and communication scope

All methods rebuild `[1,X,X]` and `[0,X,0]` and evaluate the same log cumulative-
hazard contrast. Each raw covariate is permuted five times; mean absolute
contrast change defines its importance. Recovery reports AUROC, expected Top-2
precision, and the **probability of selecting both true modifiers**. Uniform
boundary-tie handling prevents an all-zero ranking from receiving perfect
recovery merely because the signals occupy the first two columns. Contrast
RMSE uses evaluator-only truth.

The independent synthetic test population is a research evaluator. The one-
upload claim applies to FedFisher fitting and its local PV preprocessing.
Distributing the aggregate model and obtaining feature scores on private EHR
evaluation rows would need an explicit additional communication contract.
This implementation does not label that full EHR workflow as one round.

Fit, preparation and evaluation durations are recorded separately. Fit time
includes response construction, all method-specific selection/refitting,
Fisher computation, serialization and server aggregation. Actual transport,
queueing, top-level module loading and artifact I/O are excluded. Lazy framework
initialization inside fitting is included in its measured time. Data preparation
also includes site-level observable PV diagnostics; fitting recomputes its own
training/CV responses, and both costs remain visible. FedFisher reports actual
serialized upload sizes; dsLasso reports numeric-summary byte accounting,
explicitly excluding serialization overhead. Process peak RSS is a Linux
process-lifetime high-water observation, not a scheduler/cgroup reservation.

## Execution and source layout

The sole execution route is `scripts/fedfisher_server_scheduler_adapter.py`.
The additive review proposal is under
`effect_modifier_experiment/server_scheduler/survival-v1/`. The strict pilot
task `effect-modifier-survival-pilot` accepts only `{cell,scenario}` and fixes
seed 0: five methods times two scenarios. The formal task
`effect-modifier-survival-shard` retains these ten method/scenario combinations
and fixes seeds 0 through 299, partitioned by `seed % shard_count == shard_index`.
Allowed shard counts are 8 or 16; the default is 160 requests covering 3,000
scientific cells. Sharding does not change the scientific run manifest or
the data-generating process. Pilot and formal result namespaces are distinct.
There is no standalone scientific launcher. Source examples are not submitted
requests. The proposed resource profile requires operator review and pilot
measurement.

`scripts/prepare_survival_requests.py` publishes the ten pilot requests only
from a clean frozen source tree. Exact request bytes, scientific configurations,
reviewed files and source provenance are bound into an inventory, written last.
The nested outbox bundle is for operator review and is not central submission.
The generation number changes job identifiers; it does not repair an invalid
scientific namespace or authorize overwriting failed output.

After the pilot finishes, `scripts/validate_survival_pilot.py` validates all ten
scientific cells, actual successful central executions and producer attempts,
artifact/log evidence and paired invariants. Its immutable acceptance record
uses a canonical path derived from the source provenance. Method-specific
timings, observed memory and quality diagnostics support operator review;
acceptance is not a claim of good recovery or clinical validity.

Formal request publication and formal execution both require that acceptance
record and revalidate its underlying evidence against the same frozen source.
The formal task proposal is not permission to admit a large cohort before
pilot review. Prepare the formal requests with `--cohort formal` and
`--pilot-acceptance` only after this gate passes. The implementation and gate
are frozen together before the first pilot so the pilot-to-formal transition
does not require a source edit.

`data.py`, `pseudo.py`, `links.py`, `dslasso.py`, `fisher.py` and `evaluation.py`
separate scientific components. `protocol.py`, `runner.py` and `artifacts.py`
bind fixed recipes, the foreground scheduler boundary and immutable completion
evidence; `acceptance.py` owns the survival pilot's scientific checks and paired
invariants. Generic path/JSON/publication, source-provenance, request-bundle and
central execution-evidence operations come from
[`infrastructure/`](../infrastructure/README.md), which has no scientific or
scheduler-API dependencies. This shared layer does not choose the survival
estimand, fitting recipe or acceptance criteria. The active handoff and
validation evidence belong in
[`docs/codex/SCHEDULER_HANDOFF.md`](../../docs/codex/SCHEDULER_HANDOFF.md).

Durable scientific output is restricted to
`/data/del6500/FedFisher/effect_modifier_experiment/outputs/survival_v1_*_pilot/`
and `survival_v1_*_300seeds/` under the same output root. Durable acceptance
reports belong under
`/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-v1/`.
Development fixtures and caches use
`/scr/del6500/FedFisher/survival-validation/`. A clean, finalized source version
is mandatory for a pilot. Partial output, changed source, invalid semantics or
artifact hashes fail closed without overwrite; a zero process exit alone is
not scientific acceptance.

Near-zero/one predicted survival weakens both gradients and working Fisher.
The initialization ridge also shrinks weakly identified directions; local
parameter alignment and KFAC quality under site heterogeneity are empirical
questions. Low training loss or solver stationarity alone cannot establish
accurate treatment contrasts.

Before expansion, pilot review must examine PV ranges, censoring and support
at tau, modifier recovery and contrast RMSE,
solver stationarity, same-scenario paired data fingerprints, complete timing,
and communication scope. Passing software fixtures is not a substitute for
these scientific checks or the later real-EHR study.
