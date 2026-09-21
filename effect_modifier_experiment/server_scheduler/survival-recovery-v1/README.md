# Survival dsLasso recovery admission

This disabled additive proposal uses the existing foreground adapter. The
original survival task names, schemas, requests and output contracts remain
unchanged. The scientific definitions and completion gates live in
`effect_modifier_experiment/survival/recovery.py`.

The diagnostic task accepts exactly `case` and `recipe`. The schema enumerates
27 predetermined scenario/seed cases and four fixed optimization recipes:
baseline, budget only, step only and combined budget/step. It allows no free
seed, tolerance, optimizer budget, output path or resource override. There are
108 independently recoverable diagnostic jobs.

An expected iteration-limit observation is eligible for diagnostic completion
only after its saved numerical evidence validates. It is not a converged model.
Unexpected runtime or numerical errors remain process failures. A candidate
must complete all required cases under unchanged stationarity checks before
it can authorize formal work. Every formal cell uses the same globally selected
recipe, derived from immutable same-source acceptance rather than caller input.

The recovery shard task accepts exactly `scenario`, `shard_index` and
`shard_count`. Seeds remain 0 through 299, assigned by remainder under 8 or 16
shards. The default 16 shards per scenario gives 32 requests and 600 new cells.
The formal builder and runtime both enforce the diagnostic gate.

The proposed profile is CPU-only, one core and 8192 MiB. An operator must review
the increased-budget runtime and memory evidence before formal admission.
Normal requests omit resources and execution profile. The two small examples
illustrate syntax only; they are not published requests or submitted jobs.

Keep the finalized source unchanged throughout diagnostic execution, acceptance
and the gated formal cohort. Only a ServerScheduler operator may register,
validate or submit these tasks. No project-side central CLI or state mutation is
part of this proposal.
