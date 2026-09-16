# Measured CPU FedFisher reruns

This separate recipe measures diagonal/KFAC FedFisher on one CPU core, alongside
the existing CPU comparison methods. Each algorithm is fitted independently and
computes its selected Fisher representation only. A timing observation belongs
to the new model and execution; it does not fill a missing field in an old GPU
result. The original `*_factorial_90700c2_300seeds` directories remain unchanged.

The full scope is four method/architecture cells (Diag/KFAC × shallow/deep),
four original factorial variants, IID/non-IID, and seeds 0–299: 9,600 units.
The initial pilot fixes seed 0: 16 scheduler jobs and 32 scientific units.
The already measured matched one-shot FedAvg baseline is not rerun.

## Scientific recipe and differences from the historical execution

The original `SyntheticEffectModifier` generator, 10 unequal sites, 100 raw
features, 10,000-row common test set and four factorial factors are retained.
The data wrapper returns the generator's original global training tensor, so
the original `RandomState(3)` selection of 500 training rows retains its row
order. These rows also participate in local training, as in the historical
algorithm; they are not an independent validation set.

Local SGD retains 30 epochs, batch size 64, learning rate .01 and momentum .9.
Original sample-size Fisher weighting, quantization/compression and original
server routines are reused. The server takes 2,000 updates with step size .01,
checks training-subset accuracy every 100 updates, and retains the original
strict-improvement selection. Fixed iteration completion is not convergence.
KFAC's additional Fisher construction inside compression is actually executed
and charged to aggregation.

The old GPU program computed both Fisher representations between every pair
of local fits. Removing an unused representation would otherwise change later
SGD random draws. The new explicit CPU recipe isolates Fisher draws from the
training RNG and restores caller RNG state. Diag and KFAC must therefore have
identical local-model fingerprints for a given seed/data/architecture. This
does not claim bitwise reproduction of the historical GPU training trajectory.

Evaluation uses the existing CPU comparison's common probability/logit raw-X
interventions, five permutations, and expected Top-2 recovery under ties.
Probability-scale recovery is primary; logit recovery remains separately
reported. This differs from the old GPU evaluation implementation and allows
new runtime observations to share the current comparison's evaluation scope.
Test labels never enter fitting or server model selection.

## Measured time

The common `MethodTimer` records `perf_counter` wall seconds and `process_time`
CPU seconds. Stages are data preparation, training, tuning, aggregation and
evaluation. Training includes initialization, validation-input checks, local
SGD and only the selected Fisher; SGD/Fisher components are nested detail.
Aggregation includes original compression, server optimization and all
validation selection. The separate tuning bucket is unmeasured because its
work belongs to aggregation. There is no final refit in this original recipe.

Full fit time is train + tuning + aggregation, with no nested component double
counting. Evaluation and data preparation are separate. Imports, queue delay,
artifact I/O and real transport are excluded. Timings describe a serial CPU
simulation, not parallel client/network latency. Logical one-shot training
rounds are separate from server iterations; serialized wire bytes are not
invented for the original in-process aggregation.

## Artifacts and rollout

Data live under `/data/del6500/FedFisher/effect_modifier_experiment/outputs/` in
`fedfisher_timing_{cell}_{variant}_v1_{pilot|300seeds}`. Each run has a format-v2
manifest, and each seed/split has started, model, result and done files. Current
clean source, exact configuration, timer semantics, model hashes and completion
hashes are checked before success or resume. Partial artifacts fail closed.

Only the existing foreground scheduler adapter may execute the two new fixed
task contracts. Disabled proposals and schema/profile fixtures are under
`effect_modifier_experiment/server_scheduler/fedfisher-timing-v1/`.
`scripts/prepare_fedfisher_timing_requests.py` publishes the bounded pilot to a
nested project outbox after source freeze. Files in that directory are not
central submissions. The operator must review and register the new task.

After all pilot jobs complete, `scripts/validate_fedfisher_timing_pilot.py`
performs read-only semantic, central-completion and paired checks and writes a
new acceptance proof. The formal request builder revalidates that proof and
all its evidence against the same clean source before publishing 16-way shards
(256 jobs, exactly 9,600 units). Operator runtime/memory review and formal
admission remain required. No full batch is published before pilot acceptance.

`scripts/report_fedfisher_timing.py --cohort pilot|formal --output-dir PATH`
exports only fully validated completed observations to per-seed and mean/SE
tables. Fit/evaluation/data fields match the existing CPU time-cost table names;
provenance and new run IDs remain explicit. It performs no fitting or inference
and does not insert new timings into historical GPU observations.
