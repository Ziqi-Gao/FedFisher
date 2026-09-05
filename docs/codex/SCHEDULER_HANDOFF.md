# FedFisher ServerScheduler current handoff

This is the only project document for changing ServerScheduler migration facts.
Read it after `AGENTS.md`, `PROJECT_CONTEXT.md`, and `SCHEDULER_RUNBOOK.md`, then
verify every live claim against Git, scientific manifests, and the scheduler's
read-only durable state. The older migration inventory is historical only.

## Snapshot

- Recorded: 2026-09-04 22:33 CDT.
- The reviewed v2r2 implementation was committed as
  `95c7460c9251d2b35f6b91da1024ca84fb3fbd89` on `main`, eight commits ahead of
  `origin/main`.
- The eight pilot requests below were generated from that clean HEAD. This
  handoff update is the only subsequent worktree change and must be committed
  before an operator submits a pilot.
- The installed central registration was inspected read-only at
  `/home/del6500/projects/ServerScheduler/config/projects/fedfisher.toml`. It
  remains enabled with status `cpu-pilot-enabled-gpu-blocked` and contains only
  the two fixed CPU tasks below. This project session did not modify it.
- Read-only durable scheduler state still contains 68 FedFisher jobs: 61
  completed, 7 failed, and none pending or running. The latest recorded update
  is `2026-09-04T04:19:14Z`. Historical terminal states do not validate the new
  source epoch.

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
from clean HEAD `95c7460c9251d2b35f6b91da1024ca84fb3fbd89`. Their common state is
`submitted=false`; read-only central durable-state inspection found zero
matching job IDs. File presence is not submission evidence.

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

An outbox file is not submitted. A ServerScheduler operator must validate and
submit the pilots. Only after their central completion, current-source artifact
validation, and paired invariants pass may the 72 disjoint remaining requests
be generated and handed to the operator.

The outbox still contains 70 historical files. Sixty-eight have matching
durable job records. These two old v2r1 files have no matching durable record
and remain outbox-only; do not edit, submit as current work, or infer approval
from them:

- `/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-control-stationarity-full-scope-v2r1-pilot-shard-06-of-16.job-request.json`
- `/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-local-permutation-stationarity-full-scope-v2r1-pilot-shard-06-of-16.job-request.json`

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
blocked pending validated cutover and a separate exact deletion authorization.

## Blocking sequence

1. Commit this handoff-only worktree change so the project is clean.
2. Have a ServerScheduler operator validate and submit the eight pilot requests. This
   project session must not call central validate/submit or restart services.
3. After central termination, validate every pilot artifact and paired
   invariant against the same source commit.
4. Generate the 72 remaining requests, then hand them to the operator. Their
   existence in the outbox will still not mean they were submitted.

## External-action record

No central or GPU mutation occurred. This work did not edit ServerScheduler,
submit/withdraw a job, change registration state, restart a service, signal a
process, query a GPU, or write durable scientific results. It created eight
unsubmitted request files in the project outbox, ran bounded CPU regressions
and tests inside the project and FedFisher scratch roots, and created
`/scr/del6500/FedFisher/scheduler-regression/seed-285-maxiter/` for test temp and
cache redirection. One legacy test attempted to start a disposable Screen
session, but GNU Screen rejected it before creation because `/run/screen` has
the wrong mode; the test was then made source-only. A separate `screen -ls`
read-only check reported the same mode error. All central inspection remained
read-only, and the outbox files have not been centrally validated or submitted.
