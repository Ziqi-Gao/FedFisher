# Survival pilot admission review package

This disabled, additive proposal defines `effect-modifier-survival-pilot` and
`effect-modifier-survival-shard`. Pilot requests have exactly `cell` and
`scenario`, with fixed seed 0: five methods times two scenarios is ten cells.
Formal requests additionally require `shard_index` and `shard_count` (8 or 16).
Each shard processes only seeds in `range(shard_index,300,shard_count)`.
The default 16-way cohort is 160 requests and 3,000 method/scenario/seed cells.
There is no standalone local execution entrypoint.

`examples/` and `formal-examples/` contain repository examples only. These are not published outbox
requests and have not been submitted. Do not copy the additive registration
over the central registration or alter any previously registered task.

Before any computation, finalize the scientific source and configuration,
obtain a clean canonical commit, complete numerical and adapter fixtures, and
have a ServerScheduler operator review the task schema and provisional profile.
The one-core/8192-MiB/86400-second envelope is unmeasured. Operator registration,
bounded pilot admission, actual runtime/memory review and scientific acceptance
are separate gates. Admit only pilot work first. Formal execution independently
revalidates the same-source pilot proof before creating output or computing.
Its source-bound canonical proof lives under the FedFisher durable reports root;
task parameters cannot substitute a proof, arbitrary path or bypass flag.
No code fixture establishes clinical validity or a successful scientific pilot.

The sole foreground execution path is
`scripts/fedfisher_server_scheduler_adapter.py --job-manifest ...`. Its existing
protocol-v2 environment/allocation checks apply unchanged. Result namespaces
are `survival_v1_<cell>_<scenario>_pilot` and
`survival_v1_<cell>_<scenario>_300seeds`, respectively; immutable current-source, semantic,
artifact-hash and completion-marker validation govern retries. Partial or
invalid output is preserved and rejected, never overwritten. Each formal
method/scenario has one shard-independent manifest and separate seed cells;
the formal started receipt binds the validated pilot-proof digest and source.
Pilot and formal seed-zero artifacts remain in separate namespaces.

The scientific design is documented in `../../survival/README.md`. Local
pseudo-observation construction and serialized neural fitting uploads do not
constitute an end-to-end one-round EHR deployment; evaluation is an explicitly
separate simulation stage. dsLasso is an iterative comparison method.
