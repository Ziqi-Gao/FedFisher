# Ten-seed survival method comparison

This additive bounded experiment fits only **FedAvg**, restores accepted r3
**dsLasso**, and compares them with the four accepted MLP8 FedFisher/pooled
controls. It owns seeds **0 through 9**, paired across IID and covariate shift.
There are twenty independent seed/scenario requests, twenty new FedAvg models,
twenty restored linear models, and eighty referenced baseline models. There is
no 300-seed expansion or new one-shot FedAvg arm.

## Transferred FedAvg recipe

The prior single-hidden-layer eight-neuron binary experiments tied at perfect
Top-2/AUROC under the original signal strength 2. Among these tied recipes,
`fedavg_small_e5_budget30` had the shortest observed mean fitting time. We
transfer its operative hyperparameters: **5 local epochs per round, 6 rounds,
30 total local epochs**, SGD learning rate 0.01, momentum 0.9, batch size 64,
sample-size weighted parameter averaging, and fresh momentum every round.
Every round starts from the previous global aggregate. Prior loader seed offset
6200037 and the fixed-budget hash stream are retained. No new tuning, holdout
selection, or refit is introduced. This is fixed-budget completion, not a
convergence claim. The prior binary optimum is not a demonstrated survival
optimum; weaker-signal results can favor other recipes.

The survival network remains float64 **201 -> 8 ReLU -> 1**. Its output is
eta = log(-log(S)); the inverse survival mean is S = exp(-exp(eta)). It fits
untruncated hospital-local KM/IJ pseudo-values with half MSE. The binary
two-class head, cross-entropy and float32 arithmetic are not used. Data,
tau=3, independent evaluation sample, public initialization and pseudo-value
implementation are identical to the accepted survival MLP8 experiment.

## Historical models and evaluation

Accepted r3 dsLasso coefficients, strict numerical diagnostics, original
five-repeat modifier scores, CV/refit cost and iterative communication remain
bound to their original producer. We restore the model and calculate its missing
arm-wise survival predictions. We do not refit it or relabel it as a newly
trained model. The Python dsLasso-style survival adapter is not official
R/DataSHIELD software and contains no doubly robust correction.

The four accepted MLP8 controls are loaded from hash-bound V2 results without
model inference. Dataset/truth, full test X/A, pseudo-value hashes and public
initialization must match before any new evaluation. All six methods are
compared on the same ten seeds, rather than mixing a ten-seed FedAvg mean with
a 300-seed baseline mean.

Evaluation first saves control/treatment survival predictions, RMSE/MAE/bias
and calibration against simulation truth. It then reports eta(1,X)-eta(0,X)
contrast RMSE and the unchanged five-repeat raw-X permutation Top-2/AUROC.
Top-2 is recovery of both X1/X2 with uniform exact-boundary tie handling.
dsLasso discovery is reused only after exact saved baseline/hash/score checks.
New FedAvg rankings use the original evaluator and common permutation streams.

Fitting epochs, updates, communication phases and bytes are reported separately.
FedAvg has six upload phases and six model broadcasts including the first
public initialization; five broadcasts depend on fitted outcomes. Its wire
counts are actual serialized count/vector bytes without network framing.
dsLasso retains the existing numeric-summary accounting. These different
accounting scopes must remain explicit. Replayed dsLasso fitting cost is charged
separately from new evaluation time; saved-model reading is never presented as
new fitting time. Timing excludes queueing, imports, artifact I/O and transport.

## Modules and execution

- `protocol.py` fixes the small cohort and recipe; admission stays numerical-import free.
- `fedavg.py` reuses existing MLP8 local training and implements actual rebroadcasts.
- `inputs.py` checks frozen models, sources, producers and prior selection evidence.
- `computation.py` composes fitting and evaluation without filesystem ownership.
- `artifacts.py` reuses the original immutable store and validates saved arithmetic.
- `runner.py` processes only the requested seed in the foreground.
- `requests.py` and shared publication compose the twenty scientific-only requests.
- `acceptance.py` checks every saved unit, actual producer and paired invariant.

Only `effect-modifier-survival-comparators-pilot` through the canonical scheduler
adapter may execute the scientific work. The proposal below
`server_scheduler/survival-comparators-v1/` is disabled. Its one-core/8192-MiB
profile and 1800-second fallback are provisional operator review inputs, not
observed runtime. The operator must review, register and admit this bounded
cohort; fixture success and an outbox file are not submitted-job evidence.

`scripts/prepare_survival_comparators_requests.py` publishes the immutable nested
outbox bundle and inventory last. Keep that clean source frozen through all
twenty units and acceptance. Validate with
`scripts/validate_survival_comparators.py --inventory ABSOLUTE_INVENTORY`.
Saved-only acceptance never fits, generates data, predicts or replays permutations.
Partial, failed, altered or different-source outputs cannot be overwritten.

Durable results are under
`/data/del6500/FedFisher/effect_modifier_experiment/outputs/survival_comparators_v1_<scenario>_10seeds/`.
Proofs are under the durable `reports/survival-comparators-v1/` directory.
Request bundles, fixtures, logs and caches remain below `/scr/del6500/FedFisher/`.
