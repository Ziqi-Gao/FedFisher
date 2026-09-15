# Seed-zero checkpoint replay diagnostics

This is a bounded diagnostic extension of the accepted one-shot-effects pilot.
It does not change FedFisher, DR-Sieve, the DGP, the original validation rule or
the original two .25 linearization gates. The first cohort is exactly:

| Recipe | Dataset | Saved sources | Tasks |
| --- | --- | --- | --- |
| fisher_shallow / fisher_deep | scmv2_base | matching aggregation + linearity pilots | 2 |
| fisher_shallow / fisher_deep | legacy_control_iid | matching aggregation + linearity pilots | 2 |
| dr_path | scmv2_base | dr_linear + dr_spline pilots | 1 |

All tasks use original seed 0. There is no training task, compression recipe,
new seed, nuisance refit, optimizer continuation or lambda reselection in this
cohort. `protocol.py` is the strict configuration authority.

## Inputs and reproducibility

`source_bindings.json` anchors ten original units and their 150 files to the
completed scientific acceptance report. Source loading verifies every original
manifest, started/result/done record, checkpoint and site payload before
restoration, then checks saved summary arithmetic and paired inputs/models.
The original accepted source provenance and the new diagnostic source provenance
are separate. The original artifact store's current-source checks are unchanged.

Data are regenerated from the original manifest's frozen configuration. Before
any diagnostic forward or gradient evaluation, all observed T/V/H values and
IDs, split assignments, target weights, common test rows, raw test X and original
five-repeat H/external permutation fingerprints must agree exactly. A mismatch
preserves failure evidence and stops the task without overwriting any original
or partial result.

The old observed-row fingerprints did not separately cover evaluator test-site
IDs or truth arrays. The replay also pins generation source files to their
original Git bytes, regenerates a reference through that frozen SCM mechanism,
and checks test-site assignment, propensity, potential-outcome means, CATE,
modifier mask and variable roles. These additional hashes are explicitly new
replay evidence; they are not described as hashes saved by the original pilot.
Legacy has neither a fabricated test-site assignment nor CATE truth.

## Fisher diagnostics

For each site, evaluate the fixed path
`theta_s + t*(theta_F-theta_s)` with
`t = [0,.001,.01,.03,.1,.25,.5,.75,1]`.
The 32 H output rows are sampled without replacement before looking at
predictions, using `SeedSequence([0,8600021,site_id])`, and sorted only for
evaluation. Random parameter directions use stream 8700023; sampling is paired
across architectures. Interventions use the original full-H donors, even when
the scored output rows are a subset; the 32 rows are not independently shuffled.

The implementation audit uses both the normalized actual aggregation direction
and a fixed normalized Gaussian direction. At steps `1e-4,1e-5,1e-6`, compare
autograd to central differences for control and treated binary log-odds and
probability CATE, and verify the t=0 identity. Class logit *differences* are used,
so applying sigmoid reproduces the binary model probability. Record baseline
and plus/minus ReLU sign changes or exact zeros, excluding those boundary cases
from a code-error decision. These endpoint observations are not a proof that
every interior point is in a single activation region. On the remaining entries,
the fixed check is `abs(error) <= 1e-7 + 1e-4*abs(autograd)`; t=0 tolerance is
`1e-12`. Stable failures or absent usable checks block later method work.
The same audit compares the probability-effect derivative from direct autograd
with the sigmoid chain rule applied to the two log-odds derivatives (tolerance
`1e-12`), without an additional gradient pass.

Local directional derivatives of the two log-odds are computed once per sampled
original/intervened row and reused across all t. Compare their exact path values,
direct CATE Taylor expansion, and the diagnostic difference of two exact
sigmoids applied to linearly extrapolated log-odds. The sigmoid alternative is
not a quadratic matrix summary and supplies no new one-upload scoring claim.

Each point records log-odds/effect RMSE, error mean and centered RMSE, half squared
perturbation scores, ranking correlation, tie-aware Top-2 agreement, per-layer
activation changes and probability saturation. Site summaries retain original
target weights. Original-only and original-plus-intervention error moments are
distinguished. At t=1 a separate full-H, five-permutation calculation reproduces
the original full-gradient diagnostic and both .25 gates, using rtol 1e-9 and
atol 1e-10 for replay equality. Sample curves do not replace this full-H check.

SCMV2 quality comparisons use the common target-weighted independent test set:
zero CATE, each saved local model, target-weighted prediction averaging,
parameter FedAvg, strict FedFisher, the saved legacy validation-selected model
and saved exact solution of the legacy uploaded objective. Prediction averaging
and parameter averaging have different names and definitions. Record CATE RMSE,
mean bias, centered error, estimated and true heterogeneity; each local model
also has observed-outcome T/V/H cross-entropy and Brier score. Legacy reports
prediction/generalization and approximation diagnostics without CATE RMSE.

## DR path diagnostics

The single DR job reads both saved 30-candidate coefficient paths and T/V/H
summaries. It evaluates every original candidate's V risk and gap, nonzero group
count, per-variable coefficient block/norm, H importance/rank and test CATE
RMSE. The test basis is evaluated once and reused across the saved coefficients.
Original selected coefficients, lambda and index remain unchanged under the
original V-risk tie rule. Nonzero means nonzero coefficient norm in the stored
solution; it is not a significance threshold. Risk gaps and curve shape describe
whether V prefers complexity or is nearly flat; no standard error or FDR claim
is inferred from insufficient summaries. Test/H curves never select lambda.

## Conditional next batch and progression

Only after inspecting these five results may a separate plan add at most four
SCMV2-base training diagnostics: pooled shallow/deep using the union of T, and
one shallow/deep local training trajectory with fixed epochs 5,10,20,30,60.
Pooled diagnostics must report sample passes and optimizer updates. Trajectories
restart from the original initialization and complete original SGD state; they
must reproduce the original local epoch-30 parameter fingerprints. Continuing
from weights while discarding momentum is not the original trajectory. Fisher,
FedAvg, strict FedFisher and approximation checks at those checkpoints require
separate reviewed recipes and scheduler admission. None is built or requested
by this first cohort, and test curves cannot choose a reported best epoch.

Progression requires implementation checks, both original full-H .25 limits,
and no clear CATE degradation caused by trivializing the model. Freeze the final
training/selection rules before paired confirmation on new seeds. The phrase
"no clear CATE degradation" requires scientific review; this pilot does not
invent a numerical acceptance cutoff. If accurate effects and reliable local
Taylor scores cannot coexist, stop this summary route and redesign scoring
while retaining FedFisher aggregation. Fewer epochs or more damping alone are
not an innovation claim.

## Operation, outputs and validation

Only task `effect-modifier-effects-replay-pilot` in the sole project foreground
adapter can run these recipes. The separately disabled proposal and exact schema
are under `../../server_scheduler/effects-replay-v1/`. The request builder writes
exactly five normal protocol-v2 requests into a nested project outbox directory;
source and original inputs are rechecked before the final inventory is published.
An operator reviews registration, validates and submits that exact inventory.
Outbox creation is not central submission.

New results live in
`/data/del6500/FedFisher/effect_modifier_experiment/outputs/ose_replay_v1_*_pilot/`.
Each manifest binds source/configuration and original inputs. A seed unit contains
only `started.json`, `result.json`, `done.json`; no newly trained checkpoint or
deployable upload is produced. Validation recomputes summary arithmetic and
checks observational consistency without inference. A valid failed diagnostic
gate is still a completed diagnostic, never authorization for expansion.
Partial/invalid units cannot be overwritten or skipped.

Wall and process CPU measurements distinguish source restoration/data replay,
gradient audit, path evaluation, full-H reproduction and quality/path diagnostics.
Train/tuning/aggregation stages remain explicitly unmeasured with zero calls.
The process RSS high-water sample is labeled by its measurement point; it is not
a measured cgroup peak, reservation, full memory profile or deployed runtime.
Tests use tiny one-thread fixtures and scratch caches, never default-size local
experiments. Deterministic reporting reads completed artifacts and writes small
CSV/PNG files below FedFisher data/scratch, without inference or selection.
