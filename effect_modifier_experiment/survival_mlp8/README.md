# Survival probability comparison with an 8-neuron MLP

This additive experiment checks predicted survival probabilities against known
simulation truth before interpreting a counterfactual treatment contrast. All
four models have the same float64 CPU network: **201 -> 8 ReLU -> 1**, with
1,625 parameters. Historical shallow/deep recipes and outputs remain separate.

## Data and estimands

The unchanged `survival/data.py` generator supplies 10 hospitals, independently
sampled sizes of 500–1,000, 100 Gaussian covariates, and an independent common
10,000-row test set. X1 and X2 affect treatment assignment, prognosis and
treatment effect; X3–X100 are noise. Event times follow Weibull proportional
hazards (shape 1.5, baseline cumulative-hazard coefficient 0.12), with independent
exponential censoring (rate 0.05). The endpoint is survival at tau = 3. IID and
site-specific mean/scale covariate shift share test covariates, test truth and
hospital sizes for each seed. Shift changes training distributions.

Inputs are `[A, X, A*X]`. The scalar output eta represents log cumulative hazard:
`g(S) = log(-log(S))`, with survival mean `S = exp(-exp(eta))`. This is the
survival-probability form of the complementary log-log link, whose sign/orientation
must be distinguished from applying `log(-log(1-p))` to event probability.
The loss is one-half mean squared error between untruncated pseudo-values and
the predicted survival mean. Labels may lie outside [0,1]; predictions remain
probabilities. There is no doubly robust adjustment.

`survival/pseudo.py` computes Kaplan–Meier infinitesimal-jackknife pseudo-values
for survival at tau. The experiment fits:

| Model | Training response | Fitting scope |
| --- | --- | --- |
| FedFisher Diag | Each hospital's own pseudo-values | Local training, one serialized upload per hospital, diagonal aggregation |
| FedFisher KFAC | The same local pseudo-values | Identical local training, one serialized upload per hospital, KFAC aggregation |
| Pooled: site-local PV | Concatenated unchanged local pseudo-values | Central training on all hospital rows |
| Pooled: pooled KM PV | Pseudo-values recomputed from pooled observed times/statuses | Central training on all hospital rows |

The first pooled control isolates fitting/aggregation while retaining labels;
the second also measures the change from pooling the pseudo-value estimator.
Neither pooled control is a federated deployment. Archived, accepted r3 dsLasso
artifacts serve only as data-fingerprint witnesses; archived neural models and
predictions are not reused.

## Training and evaluation

All models use raw inputs, a shared public initialization, SGD with learning
rate 0.01, momentum 0.9, batch size 64 and 30 epochs. Diag and KFAC use identical
site-specific shuffle seeds and produce byte-identical local parameter vectors.
Pooled training uses the existing pooled shuffle recipe. Thirty pooled epochs
process all pooled rows; their number of updates differs from one hospital's
30 epochs. This is a fixed training budget, not a convergence claim.

The working Gaussian Fisher has variance 1. It describes curvature of the
survival-mean working model, not the sampling covariance of dependent
pseudo-values. Server aggregation minimizes the sample-size weighted quadratic
around uploaded local parameters, with ridge 1e-6 toward the public initialization.
Diag has a closed-form solution. KFAC uses conjugate gradients (maximum 2,000
iterations, residual tolerance 1e-8). A nonstationary server endpoint fails
scientific validation and preserves failure diagnostics.

Primary evaluation saves treatment/control/factual survival predictions and
compares them with generator truth using RMSE, MAE, signed mean bias and
10-bin calibration. The evaluator also saves RMS prediction differences between
each FedFisher model and each pooled control. Secondary evaluation compares
`eta(1,X)-eta(0,X)` with the known log cumulative-hazard contrast. This recipe
does not perform permutation modifier ranking or claim that good contrast
accuracy follows from accurate factual predictions alone.

Fitting reports distinguish local epochs/updates, Fisher calculation,
serialization, server iterations and pooled training. Timing excludes queueing,
artifact I/O, module imports and actual network transport; evaluation lies
outside the one-upload fitting boundary. Process peak RSS is observational.

## Files and completion contract

- `protocol.py`: fixed recipes, scientific parameter admission and seed sequences.
- `methods.py`: MLP8 training, serialized Fisher uploads, aggregation and saved-fit checks.
- `computation.py`: paired data, four fits and probability evaluation.
- `artifacts.py`: immutable results and numerical/semantic saved-array validation.
- `runner.py`: foreground, deterministic scheduler lifecycle.
- `requests.py`: scientific request plans and reviewed source bindings.
- `acceptance.py`: saved-artifact, actual-producer and paired-cohort verification without inference.

Durable runs live below
`/data/del6500/FedFisher/effect_modifier_experiment/outputs/survival_mlp8_v1_<scenario>_<pilot|300seeds>/`.
Each seed saves a started receipt, result JSON, prediction/fit arrays and a
completion marker, under a source-bound format-v2 run manifest. Partial,
different-source, altered or numerically invalid artifacts fail closed without
overwrite. Canonical acceptance proofs live below the durable
`reports/survival-mlp8-v1/` directory.

## Scheduler gates

Only the canonical ServerScheduler adapter may execute this workload. The
additive proposal under `server_scheduler/survival-mlp8-v1/` is disabled;
registration and resource review belong to a ServerScheduler operator.

The pilot task `effect-modifier-survival-mlp8-pilot` accepts only `scenario`.
Two requests cover both scenarios at seeds 0 and 49: four scientific units and
16 final models (backed by 80 local Diag/KFAC fits and 8 pooled fits). The formal task
`effect-modifier-survival-mlp8-shard` additionally accepts `shard_index` and
`shard_count` (8 or 16); it covers 300 seeds per scenario, 600 units and 2,400
final models. Default formal publication has 32 deterministic shard requests.
Formal publication and execution require a freshly reverified, canonical,
same-source MLP8 pilot proof. Old pooled/shallow/deep proofs cannot open this gate.
Scientific acceptance checks correctness rather than an accuracy cutoff;
runtime/memory review and central admission remain separate operator decisions.

`scripts/prepare_survival_mlp8_requests.py` publishes immutable request bundles
only into the project scratch outbox, with the inventory written last. An outbox
file is not a submitted job. `scripts/validate_survival_mlp8.py` verifies completed
saved results and publishes a source-addressed proof. Freeze canonical source
through the pilot and its acceptance before admitting the formal cohort.
