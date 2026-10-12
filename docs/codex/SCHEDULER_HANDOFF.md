# FedFisher ServerScheduler current handoff

This is the only project document for changing ServerScheduler migration facts.
Read it after `AGENTS.md`, `PROJECT_CONTEXT.md`, and `SCHEDULER_RUNBOOK.md`, then
verify every live claim against Git, scientific manifests, and the scheduler's
read-only durable state. The older migration inventory is historical only.

## GitHub content organization — 2026-10-12 UTC

The user clarified that the remaining clutter was visible in GitHub's current
`main`, already at `a3784653f80b209ea3c7fe4cd11fd9efe94987c3`. The earlier cleanup
improved imports, ignore rules and documentation but left generated historical
tables and request fixtures mixed into the source tree.

Read-only inspection at `2026-10-12T07:35:14Z` found zero nonterminal FedFisher
jobs and no scientific run, outbox or inventory binding to `a378465`. The saved
`4732541` acceptance proof and before-edit receipt documented below remain
unchanged; no historical proof is relabeled as acceptance under the new source.

All 70 tracked historical result files (69 CSVs and one README, 74,350 bytes)
were copied and SHA-256 verified under
`/data/del6500/FedFisher/repository-archives/github-content-cleanup-20261012/`.
Only their Git index entries were removed. Every original local file remains
unchanged and ignored, and Git history is retained. The checked-in recovery
index is `docs/archives/historical-results.json`; the identical durable
`archive-index.json` has SHA-256
`dba54a02a1c1c78c143e4cd462dbaed40390d1a12e1ba35edc1bc8b068214c57`.
The historical clone was not traversed or touched.

Seventy-four protocol-v2 request examples move byte-for-byte into
`tests/fixtures/server_scheduler_requests/`, grouped by scientific recipe.
Their scientific parameters and JSON bytes are unchanged. Schemas, disabled
proposals and plans retain their original paths and production review bindings.
Three unreferenced Slurm exports move unchanged into `scripts/legacy/slurm/`.
Protected scientific source, contract-bound launchers, central admission,
artifact namespaces and agent policies are unchanged. Historical binary report
inputs are explicit archival paths, and presentation output defaults to the
durable data tree. No new scientific run, outbox request or central action is
performed by this cleanup.

Validation passed all **992 development tests** in **147.030 seconds**, with no
failures or skips. The complete log is
`/scr/del6500/FedFisher/github-content-cleanup-20261012/full-suite.log`; fixture
scratch is `/scr/del6500/FedFisher/tests/suite-gg_m4olk`. Static review passed
350 Python files under Python-3.9 syntax, 142 JSON files, 17 TOML files, 262
local documentation links and whitespace checks. The active release tree has
586 files and no tracked CSV output. Independent review verified every retained
result, all 74 request relocations, the three export moves and unchanged agent
policies/scientific implementations. The earlier 2,232-line handoff is retained
byte-for-byte in the recorded snapshot linked below.

## GitHub preparation cleanup — 2026-10-12 UTC

The user requested repository organization before a GitHub push. Before any
source edit, saved-only comparator acceptance was replayed on clean
`473254182fecfec6bc1a9538ee82ab42ebd2bae8`. All 20 requests/units and ten paired
seeds passed semantic, manifest, artifact, provenance and producer checks, and
the fresh result exactly matched the preserved acceptance proof. Model forward
calls were forbidden. Direct durable-state inspection at
`2026-10-12T07:17:43Z` found no nonterminal FedFisher jobs. These are before-edit
observations, not claims about later central state or new-source acceptance.

The source-release receipt is
`/scr/del6500/FedFisher/github-cleanup-20261012/source-release-before-edit.json`,
SHA-256 `bc5c4a9800992b118d9d149bb8ca45328b5ac0a7885e2fd4cf84498c83c192c3`.
The original proof and all scientific artifacts retain their producer bindings.

Cleanup is on `codex/github-cleanup-20261012`. It supplements project and CLI
navigation, documents reporting dependencies and Git publication scope, and
ignores local credentials, environments, caches and generated model arrays.
Three request wrappers now share the established module-forwarding convention;
the pooled report CLI also avoids changing the import search path on import.
Scientific implementations, request plans, namespaces, central admission
guards, dependency pins and agent policies are unchanged. Historical source,
launch files, results and Git history are preserved. No new scientific task,
request publication, central action or remote push was performed.

Validation passed 13 focused architecture/request-parity tests and the full
**986-test** suite in **149.710 seconds**, with no failures or skips. Static
checks covered 349 Python files, 141 JSON files, 17 TOML files, 222 local
documentation links and whitespace. The complete fixture log is
`/scr/del6500/FedFisher/github-cleanup-20261012/full-suite.log`; runner caches
and temporary fixtures are under `/scr/del6500/FedFisher/tests/` and the
existing project-scratch fixture directories. This is development validation,
not authorization to resume historical runs under changed source.


## Scientific evidence and next source epoch

Before the first cleanup, saved-only acceptance for the ten-seed survival
comparison passed at frozen source `473254182fecfec6bc1a9538ee82ab42ebd2bae8`.
The preserved proof is:

```text
/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-comparators-v1/ten-seed-560ac5bd73516a4ec8b2d8419a81fc6ef5659892494c4fddc4d071f770f5c2ff.acceptance.json
SHA-256 c9a217caca36af7e294605553ec95c2be0ab999f5b21580da979b71d41ae6b2e
```

The new organization is development work, not scientific acceptance for a new
source epoch. Historical proofs and outputs retain their original producer
bindings. Do not resume old namespaces under changed source or treat archived
request plans as current admission. Any new scientific work requires the
runbook's source-closeout, fixture, operator pilot and registration gates.

## Recorded handoff snapshot

The complete earlier handoff is preserved byte-for-byte in
[the recorded snapshot](../archives/scheduler-handoff-20261012.md), from commit
`a3784653f80b209ea3c7fe4cd11fd9efe94987c3`: 140,612 bytes, 2,232 lines, SHA-256
`4df001a570345b8c5b6bf578830b12d447babcc89fe303c2718d93daeed3e3ce`.
It includes every earlier recipe, proposal, test record, acceptance pointer and
historical remaining-action entry. Archiving the record does not establish that
all workloads it describes are complete. Check Git, scientific manifests and
read-only central state before resuming any work; those old state observations
and action lists are historical.

Keep new changing status, validation, request/proposal evidence and blockers in
this canonical handoff. The project context and runbook retain stable procedure.
