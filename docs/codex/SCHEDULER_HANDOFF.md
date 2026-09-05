# FedFisher ServerScheduler current handoff

This is the only project document for changing ServerScheduler migration facts.
Read it after `AGENTS.md`, `PROJECT_CONTEXT.md`, and `SCHEDULER_RUNBOOK.md`, but
verify every live claim against Git, scientific manifests, and the scheduler's
read-only durable state. The older
`effect_modifier_experiment/server_scheduler/migration-inventory.md` is a
historical migration record, not current authority.

## Snapshot

- Recorded: 2026-09-04 21:29 CDT.
- Project HEAD before this review:
  `ea113cc679c7972ef6929d6233fc2cee57a8aff7` on `main`, six commits ahead of
  `origin/main`.
- The worktree is intentionally dirty with the protocol-review changes listed
  below. No commit was created.
- The installed central registration was inspected read-only at
  `/home/del6500/projects/ServerScheduler/config/projects/fedfisher.toml`. It
  has `enabled = true`, status `cpu-pilot-enabled-gpu-blocked`, and only the two
  fixed CPU tasks below. This project session did not modify it.
- Read-only durable scheduler state contained 68 FedFisher jobs: 61 completed,
  7 failed, and none pending or running. The latest recorded job update was
  `2026-09-04T04:19:14Z`. These scheduler terminal states are not a substitute
  for current-source scientific artifact validation.

## Current project-side contract

Stable foreground entrypoint:

```text
/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py
```

Registered task allowlist:

1. `effect-modifier-dslassocov-shard`: fixed CPU profile, one physical core,
   fixed 4096 MiB host memory, zero GPUs, shard count 1, 2, or 4.
2. `effect-modifier-one-shot-lasso-shard`: fixed CPU profile, one physical
   core, fixed 8192 MiB host memory, zero GPUs, shard count 8 or 16.

Both accept only `variant`, `shard_index`, and `shard_count`. The adapter stays
in the foreground, uses one compute thread, processes only the deterministic
shard seed sequence, and returns success only through the existing scientific
manifest/provenance/artifact-hash/completion-marker checks.

FedFisher has no registered GPU task and no fixture or scientific evidence for
equivalent 1/2/3/4-GPU execution. It is therefore **not eligible** for
`gpu_count_policy = "scheduler"`. GPU work remains blocked; there is no fixed
GPU proposal either.

## Changes in this uncommitted review

- `AGENTS.md` now names this handoff and permanently assigns GPU count, UUID
  selection, availability observation, and resource scheduling to
  ServerScheduler. It requires allocation-driven logical CUDA ordinals and
  1/2/3/4 evidence before any scheduler-managed GPU proposal.
- `effect_modifier_experiment/AGENTS.md` now makes the historical overlay
  procedure explicitly non-executable and carries the same future GPU
  allocation contract.
- `docs/codex/SCHEDULER_RUNBOOK.md` carries the stable allocation-driven GPU
  rule, and `effect_modifier_experiment/README.md` now labels its direct GPU
  smoke command as historical and unauthorized on the current host.
- The foreground adapter now accepts the sparse request hints preserved in a
  running manifest, including a valid memory-only hint, while still requiring
  a complete final allocation that exactly matches the fixed profile. CPU jobs
  now also require
  `NVIDIA_VISIBLE_DEVICES=void` in addition to empty `CUDA_VISIBLE_DEVICES`.
- Adapter tests cover the sparse-hint and NVIDIA-isolation cases.
- The disabled project proposal now uses the protocol-v2 fixed host-memory
  envelope fields instead of the legacy single `memory_mib` field.
- The root and method-specific READMEs/agent contract now describe 4096/8192
  MiB as exact fixed profile allocations rather than open-ended minima.
- The historical migration inventory now points here for current status.

## Disabled proposal

Path:

```text
/home/del6500/projects/FedFisher/effect_modifier_experiment/server_scheduler/registration-proposal.toml
```

SHA-256 after this review:

```text
bef678694cc932b2536da04c892b9bf1fd1e59e890bdba3018ffa5de7243215f
```

The proposal has `enabled = false`, permits CPU jobs only, contains the two
fixed `cpu-1core` profiles, declares exact fixed host-memory envelopes, and
contains no GPU task. It is a project-side proposal only; it was not copied to
the central registration.

## Requests and outbox

No request was generated or edited in this review, so there is no new job ID,
outbox path, or request digest to hand off.

The two checked-in examples remain scientific-intent-only requests and omit
both `resources` and `execution_profile`:

- `effect_modifier_experiment/server_scheduler/examples/dslassocov-shard.job-request.json`
  SHA-256
  `2ea510165ef5126e323677d9bd6861e94db7c5416f98180dd7569237310a2a5e`.
- `effect_modifier_experiment/server_scheduler/examples/one-shot-lasso-shard.job-request.json`
  SHA-256
  `5923820c80ba660cdd7a76fa5510d7967b4d6aa404dff2c0c3b987f56d1d7708`.

The existing scratch outbox contains 70 historical request files: 2
dsLassoCov and 68 one-shot requests. All 70 omit `resources` and
`execution_profile` and use only the three allowed scientific parameters.
Sixty-eight have matching durable scheduler job records. These two files have
no matching durable job record and therefore remain outbox-only observations,
not submitted-job evidence:

- `/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-control-stationarity-full-scope-v2r1-pilot-shard-06-of-16.job-request.json`,
  SHA-256
  `310a5c9f6267127910c04f881c93514ed4e2940f9d2a5b0cfeabc65dd1dd65ad`.
- `/scr/del6500/FedFisher/scheduler/outbox/fedfisher-one-shot-local-permutation-stationarity-full-scope-v2r1-pilot-shard-06-of-16.job-request.json`,
  SHA-256
  `7a6d05b0380f1848fae36dec4ae813249273c2ce16f8980e2ec9b07ee5e74638`.

Both predate current HEAD. Do not edit, remove, or infer current-source approval
from them. A future request should use a new job ID and be generated from a
reviewed clean commit, then separately handed to a ServerScheduler operator.

## Verification from this review

Successful project-environment command:

```text
PYTHONDONTWRITEBYTECODE=1 /scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B -m unittest tests.test_fedfisher_server_scheduler_adapter -v
Ran 17 tests in 1.110s: OK
```

The first attempt with the system `python` ran 14 protocol-only tests but
reported three import errors because NumPy is absent from that interpreter. No
assertion failed. Re-running in the registered project environment produced
the successful result above without installing anything.

The current ServerScheduler loader was invoked as a side-effect-free parser,
not through a scheduler CLI. It accepted the disabled proposal and both request
examples, reporting `resources=None` and `requested_profile=None` for each
example. No-cache AST parsing of the two changed Python files and
`git diff --check` also passed.

No scientific workload, GPU probe, GPU kernel, pilot, or full experiment was
run. Current-source scientific completion was not asserted.

## Remaining legacy local scheduling surfaces

These executable source paths remain for audit and are not called by the
registered adapter:

1. GPU chain:
   `launch_synthetic_effect_modifier_300seeds_local.sh` ->
   `run_synthetic_effect_modifier_local_slot.sh` ->
   `run_synthetic_effect_modifier_local_worker.sh`, plus
   `status_synthetic_effect_modifier_local.sh`. It contains `nvidia-smi`,
   physical index selection, `flock`, and Screen fan-out.
2. dsLassoCov CPU chain:
   `launch_effect_modifier_dslassocov_cpu.sh`,
   `effect_modifier_dslassocov_cpu_lock.sh`,
   `run_effect_modifier_dslassocov_cpu_worker.sh`, and
   `status_effect_modifier_dslassocov_cpu.sh`.
3. One-shot CPU chain:
   `launch_effect_modifier_one_shot_lasso_cpu.sh`,
   `effect_modifier_one_shot_cpu_lock.sh`,
   `run_effect_modifier_one_shot_lasso_cpu_worker.sh`, and
   `status_effect_modifier_one_shot_lasso_cpu.sh`.

Scratch retains 4 GPU-lock files, 1,213 dsLassoCov CPU-lock files, and 74
one-shot CPU-lock files. File presence is not a live lease. The scheduler
scratch root also retains the historical
`monitor_formal_one_shot.py`, a PID file, and an empty event log.

The 2026-09-04 read-only inspection found no matching FedFisher
worker/manager/runner/monitor process, no held FedFisher file lock, and no
FedFisher Screen socket. The only callers of the legacy scheduling helpers are
their preserved internal chains, legacy-oriented tests, and documentation; the
central adapter has no reference to them.

Deletion remains blocked until an exact user-approved retirement review after
validated cutover. Do not remove scientific workers, manifests, results,
completion markers, or historical logs as scheduler cleanup.

## Blockers and next approvals

1. Review and commit this dirty project tree before any new request; scientific
   provenance checks intentionally reject an unreviewed or mismatched source
   state.
2. Any central registration change, enable/disable action, service restart,
   validation, submission, or pilot belongs to a ServerScheduler operator and
   requires separate approval. None occurred in this review.
3. GPU integration requires a new foreground task plus scientific and fixture
   evidence at 1, 2, 3, and 4 visible GPUs. Until that exists, no
   scheduler-managed GPU proposal is truthful.
4. Central terminal job status must still be followed by task-specific
   current-source artifact and paired-invariant validation before results are
   called scientifically complete.
5. Retirement of the legacy local launch/lock/status surfaces and scratch lock
   trees requires a separate exact deletion approval after caller and holder
   revalidation.

## External-action record

This review made no external mutation: it did not modify the ServerScheduler
repository or central state, enable a registration, restart a service, submit
or withdraw a job, create an outbox request, signal a process, or query/alter a
GPU. External inspection was read-only.
