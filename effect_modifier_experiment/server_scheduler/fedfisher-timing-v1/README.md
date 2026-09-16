# Original FedFisher timing admission contract

This disabled additive proposal covers the original diagonal and KFAC
FedFisher methods on shallow and deep models. The user selected single-core CPU
execution for comparable timing. The new tasks retain the legacy four variants,
300-seed target and both IID/non-IID splits in their own timing namespaces.
Historical timing cannot be inferred or manufactured from saved models.

The shared foreground adapter is the only execution entrypoint. Requests carry
exactly `cell`, `variant`, `shard_index`, `shard_count`; resource selection and
admission belong to ServerScheduler. The fixed one-core, 8192-MiB CPU profiles
and 86400/604800-second runtime estimates are unmeasured review proposals. Tiny
fixtures do not establish memory suitability or authorize execution.

First review and register only `effect-modifier-fedfisher-timing-pilot`, then
admit the sixteen requests: four method/architecture cells by four variants,
seed 0, IID and non-IID, for 32 scientific cells. The pilot schema fixes shard
index/count to 0/1. Validate scientific artifacts, per-stage timing, unchanged
method definitions, paired invariants and runtime/memory suitability before
expansion. Exit zero alone is insufficient acceptance evidence.

The described `effect-modifier-fedfisher-timing-shard` task targets seeds 0–299
and both splits, 9600 scientific cells. Its allowed shard counts are 8 or 16,
with exact deterministic modulo partitions. Same-source complete pilot
acceptance and operator review are prerequisites. The request builder exposes
formal publication only with `--cohort formal --pilot-acceptance ABSOLUTE_JSON`.
It revalidates the complete acceptance proof and requires exact current clean
source provenance before and after publication. Formal sharding defaults to 16;
`--shard-count 8` selects the other reviewed deterministic partition. A boolean
confirmation cannot replace acceptance evidence. No formal request is produced
in this implementation phase. `timing-plan.json` records the complete bounded
scope and expansion gate.

`scripts/prepare_fedfisher_timing_requests.py --generation 1` publishes a clean,
immutable sixteen-request review bundle below the nested project outbox and
writes its inventory last. It binds the canonical source and full scientific
configuration; requests omit resources and profiles. Source examples here are
not submitted requests. Partial bundles cannot be overwritten or submitted.

Keep the source frozen from publication through execution and scientific
acceptance. New source or incompatible/partial output requires new job IDs and
appropriate independent namespaces. Never overwrite old results, weaken source
checks or resume under mismatched provenance. Operator submission remains a
separate central action; the project does not modify central state or registration.
