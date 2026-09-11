# FedFisher ServerScheduler current handoff

This is the only project document for changing ServerScheduler migration facts.
Read it after `AGENTS.md`, `PROJECT_CONTEXT.md`, and `SCHEDULER_RUNBOOK.md`, then
verify every live claim against Git, scientific manifests, and the scheduler's
read-only durable state. The older migration inventory is historical only.

## Comparison pilot request handoff — 2026-09-10

The user subsequently instructed the project to send the experiment requests.
The authorized next computation is the bounded comparison pilot; full
300-seed requests remain behind central and scientific pilot acceptance.

- Read-only inspection immediately before request preparation found the same
  148 terminal FedFisher jobs (141 completed, 7 historical failures), no
  comparison jobs, no comparison requests in outbox, and no comparison output
  namespaces. The installed central registration still listed only the two
  older tasks. It was not edited by this project task.
- The durable FedFisher intake control was also inspected read-only. It is
  armed from `2026-09-09T18:01:42Z` for the existing registered tasks. The
  current intake implementation scans only top-level outbox JSON files and
  persists rejection receipts for unregistered tasks. Do not publish these
  new-task requests into that automatic intake surface before registration.
- The complete reviewed source is being frozen on the local branch
  `codex/comparison-pilot-v1`. The pre-existing historical comparison report
  and reporting-policy edits are preserved in this source snapshot; no old
  scientific output is changed. Source publication to a remote is not part of
  this handoff.
- New builder: `scripts/prepare_comparison_pilot_requests.py`. It accepts only
  a positive generation number and creates the fixed pilot cohort from a
  clean source tree. It does not create a formal cohort or submit anything.
- Planned generation 1 contains **96 protocol-v2 requests**, exactly
  **24 scientific configurations x 4 variants**. Each request fixes
  `shard_index=0`, `shard_count=1`, and task
  `effect-modifier-comparison-pilot`; the task runs **seed 0, IID and non-IID**,
  for 192 scientific cells in total. No GPU FedFisher job is included.
- Every request contains only schema version, job ID, project, task, four
  scientific parameters, and priority. Resources and execution profile are
  omitted. No old request name or scientific namespace is reused.
- Request-phase validation passed: **30** focused request-builder, comparison
  adapter and comparison contract tests in **1.425 seconds**, including nine
  new request-publication tests. Evidence:
  `/scr/del6500/FedFisher/comparison-validation/pilot-request-tests.log`.
  The disabled proposal also passed the installed ServerScheduler's pure
  `load_registration` parser; this was a read-only parser call, not central
  registration or CLI validation. The prior 162-test implementation regression
  evidence remains recorded below.

The generation-1 delivery inventory, published only after all 96 request files
are created and revalidated, is:

```text
/scr/del6500/FedFisher/scheduler/reports/fedfisher-comparison-v1-pilot-cohort-p1.inventory.json
```

Actual request files are confined to the operator-review directory
`/scr/del6500/FedFisher/scheduler/outbox/fedfisher-comparison-v1-pilot-cohort-p1/`.
The current nonrecursive automatic intake does not admit this directory.
This is an immutable delivery bundle, not a project scheduling queue. The
immutable inventory records each
absolute path, SHA-256, job ID, parameters, configuration, expected run ID and
seed/split coverage, together with clean Git/source provenance and reviewed
source-file hashes. Its existence and validation, not this pre-generation
handoff text, establish that the complete request bundle was written. A
partial bundle without its inventory must not be submitted. An inventory with
`submitted=false` records delivery to the project outbox, not central
acceptance.

**Operator action:** inspect the disabled additive
`effect_modifier_experiment/server_scheduler/comparison-registration-proposal.toml`,
the pilot parameter schema and fixtures, and the request inventory. Review the
provisional profile, register/admit the pilot task centrally, then validate and
submit only the inventory's exact request list. Do not replace the installed
registration with the additive proposal or enable formal work prematurely.
The project session must not perform those central mutations or commands.

Immediately before submission/start, verify that canonical HEAD and clean
source provenance still equal the inventory's values. Leave that source
unchanged through pilot completion and paired scientific validation. Record
post-freeze receipts and transient observations in FedFisher scratch reports;
editing this tracked handoff during the cohort would change its provenance.
The durable central job records are the eventual acceptance/status authority.

## Method-comparison extension — 2026-09-10

The user authorized implementation of additional comparisons and timing, with
the existing prediction-difference procedure retained. This section supersedes
the implementation-status portion of the older v2r2 snapshot below; it does not
change or revalidate that frozen scientific cohort.

- Work started on `main` at `3838f0c`, with existing user changes in root
  `AGENTS.md`, `SCHEDULER_RUNBOOK.md`, and the untracked
  `scripts/make_one_shot_lasso_fedfisher_comparison.py`. Those edits were
  preserved. During the implementation phase, no commit, central registration
  change, outbox request, pilot, formal run, GPU operation, or process control
  was performed. The subsequent request phase is recorded above.
- Before source edits, read-only durable scheduler inspection found 148
  FedFisher jobs: 141 completed and 7 historical failures, with no nonterminal
  jobs. All 80 v2r2 jobs were completed, exit 0. Twelve v2r2 run manifests
  retained clean source commit `cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe` and
  provenance `8e15387dbd94c719fd761ca40a19ba2cbe272f6124f1a05d521a22a0fc442b93`.
  The earlier full scientific validation is recorded below. This already
  dirty, later checkout cannot claim a fresh current-source validation of
  those frozen outputs; they were neither resumed nor modified.
- The read-only installed central registration still contained only the two
  existing CPU tasks. New names `effect-modifier-comparison-pilot` and
  `effect-modifier-comparison-shard` are project-side contracts and a disabled
  additive proposal, not live registration or submission evidence.

### Implementation and scientific scope

`effect_modifier_experiment/causal_comparison/` contains fixed recipes, true
multiround FedAvg, shared-effect R estimators, wrappers around existing CPU
LASSO/one-shot methods, unified probability-scale evaluation, timing,
provenance-bound immutable artifacts, and completed-artifact reporting.

- No standard X-learner was added. Outcome methods retain `[A,X,A*X]` training,
  consistent treatment/control prediction differences, and raw-X intervention.
  Existing dsLassoCov is already an iterative gradient method; its objective,
  CV and stationarity rules remain intact, with optional communication
  observation covering the actual aggregation calls.
- The 24 fixed cells cover iterative dsLassoCov, local logistic LASSO, matched
  one-shot FedAvg and group-LASSO for two MLP architectures, two homogeneous
  R estimators, and E=1/5/10/30 multiround FedAvg for both architectures under
  validation-patience and fixed-total-30-epoch protocols. Selection plus full
  refit costs count in full. Safety exhaustion is not scientific completion.
- R nuisances remain local with nested training-only selection/cross-fitting.
  The server receives coefficient/precision/count summaries and imposes one
  shared effect vector. The unpenalized and adaptive-LASSO versions differ in
  the final penalty only. Sample-size scaling, shared EBIC degrees of freedom,
  constant-response handling, numerical failure rules and reference-code
  deviations are documented in the new README.
- Preserve the existing DGP and four factorial variants. The R model is a
  linear probability-effect working model on this binary, label-partitioned
  DGP. It does not inherit all assumptions of the reference paper. New common
  endpoints use probability contrasts, with logit endpoints retained only for
  outcome models. Expected Top-k recovery handles cutoff ties uniformly; old
  index-tie metrics remain diagnostics and old result metrics are unchanged.
- `main.py --record_timing` and optional `train_model.py` hooks observe the
  original Fisher path. Both Fisher computations still actually run; component
  sums are labeled estimates and the actual path is marked not directly
  comparable. These hooks do not supply a GPU execution contract or measured
  standalone FedFisher results.
- New pilot namespaces contain seed 0 and both splits; formal namespaces
  contain seeds 0..299 with deterministic 8/16-way shards. Both require clean
  finalized provenance. Partial/invalid cells fail closed without overwrite;
  retries validate manifests, scientific identity, semantics, hashes and
  markers before skipping. The original v2r2 task namespaces remain unchanged.

### Project-side review package and next gate

- Disabled proposal:
  `effect_modifier_experiment/server_scheduler/comparison-registration-proposal.toml`.
- Strict parameter schemas and unsubmitted examples:
  `comparison-{pilot,shard}-parameters.schema.json` and corresponding
  `server_scheduler/examples/comparison-*.job-request.json`.
- Sole execution path remains
  `scripts/fedfisher_server_scheduler_adapter.py`; both new tasks inherit its
  complete protocol-v2 running-manifest/environment/allocation validation.
- The draft fixed one-core, 8192 MiB, zero-GPU envelope and fallback runtimes
  are provisional review inputs, not empirical profiles. An operator must
  review/merge the disabled proposal and admit bounded pilots before any
  formal cohort. Pilots must establish runtime/memory suitability, convergence,
  all four-variant paired invariants and scientifically valid completion.
- Finalize source/configuration and commit before operator pilot requests.
  Do not reuse historical requests or output namespaces. Do not submit formal
  work merely because the code fixtures pass. The user has not been asked to
  approve a speculative central action; the concrete review package is local.

### Development validation evidence

- Full no-bytecode, one-thread CPU regression suite: **162 tests passed** in
  27.672 seconds (`python -B -m unittest discover -s tests -p 'test_*.py' -v`).
  This includes the existing 98 regressions and 64 added tests. Numerical
  fixtures are deliberately small; no full-dimension pilot was run locally.
- After the last artifact seed/site-identity binding change, all **16** focused
  comparison contract tests passed again in 0.906 seconds. They cover paired
  composition, true tiny FedAvg fitting, ties, immutable retry behavior,
  rehashed semantic tampering, partial output and symlink rejection.
- Separate earlier checks also passed: R/FedAvg/timing numerical fixtures,
  legacy output equality with/without observation, complete new-task v2
  manifest fixtures, and reporting aggregation/invariant checks.
- No-cache compilation of 23 changed/untracked Python files, strict JSON
  parsing of six new JSON files, TOML parsing of the disabled proposal, and
  `git diff --check` passed. The Python count includes the user's pre-existing
  untracked comparison-report script, which was inspected but not modified.
- The only suite warning was the previously observed NetworkX duplicate
  `nx-loopback` backend warning. There were no test failures or GPU calls.
- Logs:
  `/scr/del6500/FedFisher/comparison-validation/full-suite.log` and
  `/scr/del6500/FedFisher/comparison-validation/final-contract-tests.log`.
  All test temporary output and caches used this scratch subtree or existing
  FedFisher test scratch subdirectories. No new durable scientific result was
  generated. These checks are development evidence, not pilot acceptance or
  new-method performance results.

## Historical v2r2 snapshot

- Recorded: 2026-09-06 14:08 CDT.
- The reviewed v2r2 implementation was committed as
  `95c7460c9251d2b35f6b91da1024ca84fb3fbd89` on `main`, eight commits ahead of
  `origin/main`.
- The pilot-request handoff was committed as
  `cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe`. All v2r2 scientific manifests
  and results bind that clean commit. Final scientific validation was completed
  from a clean `codex/v2r2-run-freeze` checkout at that exact commit before the
  user restored `main` to
  `1943817dd595f6acf7e968ab042b09269efde84f`, ten commits ahead of
  `origin/main`. This handoff update is the only new worktree change.
- The installed central registration was inspected read-only at
  `/home/del6500/projects/ServerScheduler/config/projects/fedfisher.toml`. It
  remains enabled with status `cpu-pilot-enabled-gpu-blocked` and contains only
  the two fixed CPU tasks below. This project session did not modify it.
- Read-only durable scheduler state contains 148 FedFisher jobs: 141 completed,
  7 historical failures, and none pending or running. All 80 v2r2 cohort jobs
  are completed with exit code 0: 16 dsLasso shards and 64 one-shot shards.
  The latest v2r2 update is `2026-09-06T15:20:56Z`.

## Current project-side contract

The sole foreground entrypoint remains:

```text
/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py
```

The registered allowlist remains CPU-only:

1. `effect-modifier-dslassocov-shard`: fixed `cpu-1core`, fixed 4096 MiB host
   memory, zero GPUs, and shard count 1, 2, or 4.
2. `effect-modifier-one-shot-lasso-shard`: fixed `cpu-1core`, fixed 8192 MiB
   host memory, zero GPUs, and shard count 8 or 16.

Both accept only `variant`, `shard_index`, and `shard_count`. Normal requests
omit `resources` and `execution_profile`. The adapter stays in the foreground,
uses one compute thread, processes only the deterministic shard sequence, and
requires the manifest/provenance/semantic-identity/artifact-hash/completion
checks before skipping work.

FedFisher has no GPU task and no scientific or fixture evidence for equivalent
1/2/3/4-GPU execution. It is not eligible for scheduler-managed GPU sizing and
has no fixed GPU proposal. The two registered tasks truthfully remain fixed
zero-GPU CPU tasks.

## Current migration changes

- The deterministic one-shot failure was reproduced exactly at seed 285,
  non-IID, Gaussian training, site 8, CV fold 2, lambda index 27. With the old
  2000-iteration bound its proximal-gradient and KKT residuals were
  `2.0143112469904565e-05` and `2.0143112469904544e-05`, above the unchanged
  `1.4534929894602263e-05` threshold.
- Formal local-logistic runs now fix the primary optimizer limit at 4000. No
  tolerance, stationarity threshold, lambda grid, CV rule, selected-fit
  validation, or 50-iteration momentum-reset polish rule was weakened.
- The exact regression now converges without polish. The selected converged CV
  solution is index 23; its slowest selected fold takes 3020 iterations and
  every selected fold/final fit passes both original residual checks. The
  changed index is the result of evaluating the predeclared CV rule on fully
  converged candidates, not a failure-time fallback.
- New coherent output namespaces are
  `dslassocov_<variant>_server_scheduler_v2r2_300seeds` and
  `<one_shot_family>_<variant>_stationarity_4000iter_full_scope_server_scheduler_v2r2_300seeds`.
  v2/v2r1 outputs are preserved but cannot be resumed or combined because
  their manifests bind older source provenance.
- `scripts/prepare_fedfisher_scheduler_requests.py` now constructs strict
  protocol-v2 pilot or remaining request sets only from a clean Git tree,
  refuses overwrites, emits absolute paths/SHA-256/job IDs/HEAD, and marks the
  files as unsubmitted. Remaining requests require the explicit
  `--confirm-pilots-validated` gate.
- Request-set tests prove exact disjoint coverage: eight pilots plus 72
  remaining requests cover all 16 dsLasso shards (4 variants x 4) and all 64
  one-shot shards (4 variants x 16) exactly once. Every request contains only
  the three allowed scientific parameters and omits resource/profile fields.
- The old test that created a detached GNU Screen session was replaced by a
  source-level legacy FD-closure assertion. Current validation must not launch
  a local Screen scheduler merely to test preserved audit code.

## Pilot requests

Eight v2r2 generation-1 pilot requests were created under the project outbox
from clean HEAD `95c7460c9251d2b35f6b91da1024ca84fb3fbd89`, recorded in commit
`cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe`, and submitted by a
ServerScheduler operator. All eight durable jobs are now `completed`, attempt
1, exit code 0, with no failure reason.

The requested pilot coverage is:

- dsLasso: shard 0/4 for each of the four variants;
- one-shot: shard 6/16 for control and local permutation;
- one-shot: shard 13/16 for Gaussian shift and the combined arm.

Those one-shot shard choices include the preserved deterministic failure
sequences while keeping the paired control/local and Gaussian/combined arms
aligned. Exact request records are:

```text
dc39cc5c0a81a2468e15c974d993601dd6ebf8036b1852b30ab56fafdc3eb631
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-dslasso-control-v2r2-pilot-cohort-p1-shard-00-of-04.job-request.json
job_id=fedfisher-dslasso-control-v2r2-pilot-cohort-p1-shard-00-of-04

b4585c270a007932f97215aa4e49c831a87fdf7ac2f459df081bdc367c7b5fae
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-dslasso-site-gaussian-shift-v2r2-pilot-cohort-p1-shard-00-of-04.job-request.json
job_id=fedfisher-dslasso-site-gaussian-shift-v2r2-pilot-cohort-p1-shard-00-of-04

c12c2797955362e8d590b379901ba3f7cc54e461ea7ef3ee890b76a000e92d59
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-dslasso-local-permutation-v2r2-pilot-cohort-p1-shard-00-of-04.job-request.json
job_id=fedfisher-dslasso-local-permutation-v2r2-pilot-cohort-p1-shard-00-of-04

0a1168729ea1e51e2d2f2b96e43b1589f81c1fa7fa7b1695f26fad13a88dd1cb
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-dslasso-site-gaussian-local-permutation-v2r2-pilot-cohort-p1-shard-00-of-04.job-request.json
job_id=fedfisher-dslasso-site-gaussian-local-permutation-v2r2-pilot-cohort-p1-shard-00-of-04

6fc9d007c00401c18146369ede3b71d2e5c7da769704340a39db8b5df086da62
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-control-v2r2-pilot-cohort-p1-shard-06-of-16.job-request.json
job_id=fedfisher-one-shot-control-v2r2-pilot-cohort-p1-shard-06-of-16

63e81c2e95692fcec2ddbbd39c4a772f755169afbc541478443e0c6d4693be8d
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-site-gaussian-shift-v2r2-pilot-cohort-p1-shard-13-of-16.job-request.json
job_id=fedfisher-one-shot-site-gaussian-shift-v2r2-pilot-cohort-p1-shard-13-of-16

a5a204231262305e9efc3eb2c595a06d098f1541b61ed497cda44350bc95ef97
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-local-permutation-v2r2-pilot-cohort-p1-shard-06-of-16.job-request.json
job_id=fedfisher-one-shot-local-permutation-v2r2-pilot-cohort-p1-shard-06-of-16

02846cb4f517d646cf67f0870a2fcd977ad8558ccf660c1db99c2ae37542cd0d
/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-site-gaussian-local-permutation-v2r2-pilot-cohort-p1-shard-13-of-16.job-request.json
job_id=fedfisher-one-shot-site-gaussian-local-permutation-v2r2-pilot-cohort-p1-shard-13-of-16
```

The pilot scientific gate passed against clean current commit
`cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe`:

- dsLasso: four manifests, 300 seed markers, and 600 result cells validated;
  control/local, Gaussian/combined, and common-test invariants all passed.
- one-shot: eight manifests and 444 result cells validated; 222 paired
  control/local or Gaussian/combined comparisons passed for training, test,
  payload, and aggregate-model fingerprints.
- dsLasso stderr is empty. Each one-shot stderr contains only the same 228-byte
  NetworkX duplicate-backend runtime warning; there is no stationarity,
  artifact, adapter, or scheduler error.

The outbox still contains 70 historical files. Sixty-eight have matching
durable job records. These two old v2r1 files have no matching durable record
and remain outbox-only; do not edit, submit as current work, or infer approval
from them:

- `/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-control-stationarity-full-scope-v2r1-pilot-shard-06-of-16.job-request.json`
- `/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-local-permutation-stationarity-full-scope-v2r1-pilot-shard-06-of-16.job-request.json`

## Remaining requests

After the pilot gate passed, 72 v2r2 generation-1 remaining requests were
created from clean commit `cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe`:

- 12 dsLasso requests: shards 1, 2, and 3 of 4 for every variant;
- 60 one-shot requests: every shard except each variant's already completed
  pilot shard.

Together with the pilots, the 80 unique job IDs cover every dsLasso 0..3 shard
and every one-shot 0..15 shard for all four variants exactly once. Every actual
request passed the read-only ServerScheduler `JobRequest.from_file` parser,
contains only `variant`, `shard_index`, and `shard_count`, and omits both
`resources` and `execution_profile`.

A ServerScheduler operator subsequently submitted all 72 remaining requests.
Read-only durable state now records all 72 as `completed` with exit code 0: 12
dsLasso and 60 one-shot jobs. Sixty-four completed on attempt 1. Eight required
attempt 2 after attempt 1 was classified as retryable `cpu_interrupted` with
reason `scheduler and child processes are no longer alive`. In every such case,
attempt 1 had already written the complete scientific shard and attempt 2
validated the existing manifest-bound cells without overwriting them. Across
pilots and remaining work, 72 jobs used one attempt and 8 used two attempts.

The complete per-request absolute path, SHA-256, job ID, task, and parameters
are recorded in:

```text
/scr/del6500/FedFisher/scheduler/reports/fedfisher-v2r2-remaining-cohort-p1.inventory.json
SHA-256 ee8b8cb54f13c84d9963f0b74e91be0d7f11dd9102bb6dde36c4e698a9296b00
```

That inventory records generation HEAD `cbcde8f`, a clean generation tree,
`request_count=72`, and `submitted=false`. That field records generation-time
state only; the durable job records above are the later authoritative evidence
of central acceptance and completion.

## Proposal

The project-side proposal remains disabled and unchanged:

```text
/home/del6500/projects/FedFisher/effect_modifier_experiment/server_scheduler/registration-proposal.toml
SHA-256 bef678694cc932b2536da04c892b9bf1fd1e59e890bdba3018ffa5de7243215f
```

It contains only the two fixed zero-GPU CPU tasks. No central registration,
enablement, or service change is needed or authorized from this project
session for the source epoch change.

## Verification

Exact scientific regression, with one CPU thread and caches/temp redirected to
the FedFisher scratch tree:

- old 2000-iteration configuration: reproduced the canonical seed-285
  stationarity failure at the exact site/fold/lambda and residuals above;
- new 4000-iteration configuration: passed the unchanged residual criteria,
  selected index 23, slowest selected fold 3020 iterations, no polish attempt.

Full no-bytecode project suite after the changes:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/del6500/projects/FedFisher \
TMPDIR=/scr/del6500/FedFisher/scheduler-regression/seed-285-maxiter \
XDG_CACHE_HOME=/scr/del6500/FedFisher/scheduler-regression/seed-285-maxiter \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B \
-m unittest discover -s tests -p 'test_*.py' -v
Ran 98 tests in 26.769s: OK
```

`git diff --check` passed. The request CLI was also exercised against the dirty
tree and failed closed without writing an outbox file. The ServerScheduler
`JobRequest.from_dict` parser accepted all 8 in-memory pilot and 72 in-memory
remaining requests with `resources=None` and `requested_profile=None`; no
central CLI was called.

After commit `95c7460`, the 23 focused request/adapter/seed-285 tests passed in
18.355 seconds. All eight materialized requests were then accepted by the
read-only `JobRequest.from_file` parser with `resources=None` and
`requested_profile=None`; their SHA-256 digests matched the generator report,
their job IDs were unique, and their top-level and parameter fields were exact.

After pilot scientific validation and remaining-request materialization, the
22 focused request/adapter tests passed in 1.095 seconds. The complete 72-file
inventory and its digest were independently reread after creation.

Final full-cohort scientific validation was run before leaving the clean frozen
checkout at `cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe`:

- all 80 v2r2 jobs were `completed` with exit code 0; all expected attempt
  stdout/stderr logs existed;
- all 88 attempts had either empty stderr or the exact known two-line,
  228-byte NetworkX duplicate-backend warning; no unexpected stderr and no
  stdout Traceback, out-of-memory, killed, nonstationary, scientific-rejection,
  or error signature was found;
- all four dsLasso manifests matched current source provenance; the exact
  inventory of 1,200 seed markers and 2,400 result cells passed semantic,
  stationarity, artifact-hash, fit-metadata, and marker-hash validation;
- the canonical full dsLasso invariant checker passed all control/local,
  Gaussian/combined, and common-test checks and reported
  `validated_result_cells=2400`;
- all eight one-shot manifests matched current source provenance; the exact
  inventory of 2,400 logistic and 4,800 MLP cells passed semantic identity,
  communication-contract, artifact-hash, fit-metadata, and marker validation;
- the canonical one-shot checker passed representative full four-arm seed 0
  checks for both families, and the exhaustive check passed 3,600 paired-cell
  comparisons plus 1,800 four-variant common-test checks across every seed,
  split, and model.

The common validated provenance digest for all v2r2 scientific output is
`8e15387dbd94c719fd761ca40a19ba2cbe272f6124f1a05d521a22a0fc442b93`.
The complete supplemental dsLasso and one-shot experiment is scientifically
complete; central process exit alone was not used as the acceptance criterion.

## Remaining legacy local scheduling surfaces

The following remain preserved and executable in source for audit, but are not
called by the central adapter and are not authorized launch paths:

1. GPU chain:
   `launch_synthetic_effect_modifier_300seeds_local.sh` ->
   `run_synthetic_effect_modifier_local_slot.sh` ->
   `run_synthetic_effect_modifier_local_worker.sh`, plus
   `status_synthetic_effect_modifier_local.sh`; it contains physical GPU
   selection, `nvidia-smi`, `flock`, and Screen fan-out.
2. dsLasso CPU chain:
   `launch_effect_modifier_dslassocov_cpu.sh`,
   `effect_modifier_dslassocov_cpu_lock.sh`,
   `run_effect_modifier_dslassocov_cpu_worker.sh`, and
   `status_effect_modifier_dslassocov_cpu.sh`.
3. One-shot CPU chain:
   `launch_effect_modifier_one_shot_lasso_cpu.sh`,
   `effect_modifier_one_shot_cpu_lock.sh`,
   `run_effect_modifier_one_shot_lasso_cpu_worker.sh`, and
   `status_effect_modifier_one_shot_lasso_cpu.sh`.

Scratch still retains the historical lock files and monitor artifacts recorded
in the earlier handoff. File presence is not a live lease. Retirement remains
outside the completed scientific scope and requires a separate exact deletion
authorization.

## Remaining actions

1. Commit this handoff-only completion update so `main` is clean again.
2. No v2r2 supplemental computation, retry, or submission remains.
3. The preserved legacy local scheduling surfaces may be retired only under a
   separate exact authorization; their presence is not a completion blocker.

## External-action record

The 8 pilot and 72 remaining submissions and their scientific outputs were
performed by the external ServerScheduler operator/attempts, not by this
project inspection. This project work did not edit ServerScheduler,
submit/withdraw a job, change registration state, restart a service, signal a
process, or query a GPU. It created the 72 request files and one scratch audit
inventory before operator submission, and ran read-only pilot and full-cohort
artifact/invariant validation plus bounded CPU tests inside the project and
FedFisher scratch roots. The user performed the final `git switch main`; the
project session's own switch attempt failed before mutation because `.git` was
read-only. Earlier work created
`/scr/del6500/FedFisher/scheduler-regression/seed-285-maxiter/` for test temp and
cache redirection. One legacy test attempted to start a disposable Screen
session, but GNU Screen rejected it before creation because `/run/screen` has
the wrong mode; the test was then made source-only. A separate `screen -ls`
read-only check reported the same mode error. All current central inspection
remained read-only. A read-only attempt to inspect the user service/journal for
the exact cause of the eight supervisor interruptions was denied by the local
bus/journal permissions; no service action followed.
