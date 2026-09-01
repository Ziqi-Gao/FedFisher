# Legacy results policy after ServerScheduler cutover

Decision date: 2026-08-31 (America/Chicago).

The project owner selected the single-canonical-tree migration. Continuing the
source-identical pre-ServerScheduler experiments is intentionally abandoned so
that `/home/del6500/projects/FedFisher` can become the only production code
root for centrally submitted work.

## Scope and status

All experiment artifacts created before the canonical ServerScheduler cutover
are **legacy, retained, and non-authoritative**. This includes existing content
under `/data/del6500/FedFisher`, especially the incomplete formal one-shot
cohort that reported 2,960 validated cells of 7,200 at migration inventory
time, plus historical dsLassoCov and GPU effect-modifier runs.

- Do not delete, rename, overwrite, or rewrite these artifacts.
- Do not resume their missing seeds or cells.
- Do not use them as formal scientific results or combine them with
  scheduler-era output.
- Preserve their manifests, provenance, hashes, completion markers, logs, and
  status records for audit and method-development history.
- A process exit code or legacy marker does not promote an artifact back to
  authoritative status.

## Scheduler-era authority

Only output produced by an enabled, reviewed ServerScheduler task after its
pilot may become authoritative. The initial CPU adapters derive disjoint fixed
`server_scheduler_v2` run namespaces, validate current source provenance and
scientific configuration, skip only fully validated completion markers, and
refuse partial or invalid output.

GPU workloads and every other unregistered program remain blocked. Adding an
authoritative workload requires a foreground adapter path, strict parameter
schema, resource profile, fixture tests, approved pilot, and central
registration. This policy does not authorize direct launchers as a fallback.
