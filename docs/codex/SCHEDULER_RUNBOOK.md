# FedFisher ServerScheduler Runbook

This runbook describes stable protocol-v2 operating procedure. It deliberately
contains no active job IDs, queue counts, current commit binding, or completion
status. Live central registration, durable job JSON, Git, and scientific
manifests are authoritative.

`AGENTS.md` files are normative; this document is a descriptive operating
reference. If they disagree, stop and follow the applicable `AGENTS.md`.

## Ownership boundary

FedFisher owns scientific policy: allowed task parameters, deterministic seed
shards, result namespaces, manifests, provenance, semantic identity, artifact
hashes, completion markers, resume rules, and scientific acceptance.

ServerScheduler owns machine policy: admission, queue order, fairness, global
concurrency, execution-profile selection, physical CPU/NUMA/GPU allocation,
leases, retries, logs, and runtime estimates.

A FedFisher project session may create protocol-v2 request files only below:

```text
/scr/del6500/FedFisher/scheduler/outbox/
```

An outbox file is not a submitted job. Only a ServerScheduler operator may
validate and submit it. Project sessions must never write central pending,
running, history, lease, retry, audit, log, or performance state.

The only canonical foreground entrypoint is:

```text
/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py
```

It accepts only `--job-manifest ABSOLUTE_RUNNING_JSON`. It must remain in the
foreground and may not create Screen/tmux sessions, daemonize, detach, launch a
project resource scheduler, or acquire project CPU/GPU locks.

## Registered task contracts

Both tasks accept exactly three parameters and reject additional fields:

```text
variant: control | site_gaussian_shift | local_permutation |
         site_gaussian_local_permutation
shard_index: zero-based integer
shard_count: task-specific allowed integer
```

| Task | Allowed shard count | Profile | Minimum memory | GPU |
| --- | --- | --- | --- | --- |
| `effect-modifier-dslassocov-shard` | 1, 2, or 4 | fixed `cpu-1core` | 4096 MiB | 0 |
| `effect-modifier-one-shot-lasso-shard` | 8 or 16 | fixed `cpu-1core` | 8192 MiB | 0 |

For both tasks, `0 <= shard_index < shard_count`, seeds are fixed to 0 through
299, and a shard may process only seeds congruent to its index modulo the shard
count. Requests do not select seeds, paths, run IDs, methods, hyperparameters,
physical CPUs, NUMA nodes, or runtime estimates.

Registration fallback runtime values are not live estimates. The scheduler may
replace them with a persisted EWMA for the same task/profile/allocation
identity. Query the running manifest or durable state when a current estimate
is required.

GPU jobs are disabled. Legacy GPU launchers, device polling, GPU locks, and
manual `CUDA_VISIBLE_DEVICES` selection are not authorized bypasses. Any future
GPU task requires a separately reviewed foreground adapter, parameter schema,
memory/utilization/sharing/model profile, fixture suite, pilot, and central
registration change.

## Request construction and operator handoff

Normal protocol-v2 requests omit `resources` and `execution_profile` so the
scheduler selects the registered profile:

```json
{
  "schema_version": 2,
  "job_id": "stable-unique-job-id",
  "project": "FedFisher",
  "task": "effect-modifier-one-shot-lasso-shard",
  "parameters": {
    "variant": "control",
    "shard_index": 0,
    "shard_count": 16
  },
  "priority": 0
}
```

Use one request per independently recoverable shard. Job IDs must be unique and
must not be reused after failure. Parameters must not contain commands, secret
values, arbitrary environment variables, hardware choices, paths, or scientific
configuration that the registered task fixes.

The following commands are for a **ServerScheduler operator session only**, not
a FedFisher project session:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/del6500/projects/ServerScheduler/src \
/usr/bin/python3.12 -m server_scheduler validate /absolute/request.json

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/del6500/projects/ServerScheduler/src \
/usr/bin/python3.12 -m server_scheduler submit /absolute/request.json
```

Successful `submit` output is acceptance evidence. The presence of the outbox
request is not. Preserve requests for audit and do not edit an accepted request
in place.

## Adapter validation

The adapter accepts only a non-symlink, strict JSON protocol-v2 running
manifest. It validates the project and task allowlist, fixed execution profile,
requested resources, final allocation, CPU IDs, NUMA node, memory, lease,
one-based attempt, runtime estimate, and plural GPU identity arrays against the
complete scheduler-provided environment. Protocol-v1 singular GPU fields are
invalid.

CPU jobs must use `SERVER_SCHEDULER_CPU_CORES`, preserve scheduler affinity, and
run no more software workers or BLAS threads than allocated physical cores.
They must not inspect other devices or implement global concurrency.

## Live status from a project session

Use direct read-only inspection of the scheduler's durable JSON:

```bash
jq . /data/del6500/ServerScheduler/state/jobs/JOB_ID.json
jq . /data/del6500/ServerScheduler/state/retries/JOB_ID.json
```

Use each job JSON's `stdout_log` and `stderr_log` fields to locate exact attempt
logs under `/scr/del6500/ServerScheduler/logs/`. Scratch queue mirrors under
`/scr/del6500/ServerScheduler/queue/{pending,running,completed,failed}/` are
useful secondary observations; durable `/data` state is authoritative.

Do not call these commands from a FedFisher project session merely to query
state:

```text
server_scheduler validate
server_scheduler status
server_scheduler queue
server_scheduler performance
server_scheduler plan
server_scheduler doctor
server_scheduler observe
```

Their implementations can acquire and initialize central locks, reconcile
state, create missing retry state, or observe machine resources. Operators may
use the supported CLI from an authorized ServerScheduler session.

Project sessions must also never run submission or broker commands, including
`submit`, `init`, `run-once`, `dispatch`, `serve`, or any command that mutates
central state.

## At-least-once delivery and scientific completion

Scheduler delivery is at least once. A retry may start after partial scientific
output exists. Each attempt must therefore classify every owned unit as:

- `validated`: all manifest, current-source provenance, semantic identity,
  artifact hashes, and completion-marker checks pass; skip safely.
- `missing`: no conflicting output exists; run it.
- `partial` or `invalid`: fail closed without overwrite.

The dsLassoCov task validates both IID and non-IID results before creating and
revalidating a seed marker. The one-shot task validates each
seed/split/model-family cell independently and immediately revalidates any new
marker. A zero exit code is necessary but never sufficient.

Before changing tracked source after a formal cohort, verify:

1. every central job reached the expected terminal state and attempt;
2. every expected seed belongs to exactly one shard;
3. every scientific cell passes current-source manifest, semantic identity,
   artifact hash, and completion-marker validation;
4. all four-variant paired invariant checks pass;
5. stderr contains no unexplained scientific or runtime failure;
6. Git remains at the source provenance recorded by the manifests.

Run final scientific checkers before any tracked edit because they intentionally
require current source provenance to match the run manifests.

## Rollout gates

Use this sequence for every new or materially changed task epoch:

1. Review the fixed task name, strict parameter schema, profile, adapter path,
   and fixture tests.
2. Run no-cache unit, contract, and adapter tests without launching a formal
   experiment.
3. Create and centrally validate bounded pilot requests.
4. Have the operator submit the pilot.
5. Validate central termination and all scientific completion evidence.
6. Confirm paired invariants and obtain a representative runtime estimate.
7. Generate and validate the full request set with exact shard coverage and
   unique job IDs.
8. Have the operator submit the formal set.
9. Monitor and scientifically validate the entire cohort.
10. Freeze results before changing source or documentation that participates in
    provenance.

## Failure handling and rollback

On failure, inspect durable job/retry JSON and exact attempt logs read-only.
Separate scheduling failure from scientific partial/invalid output. Do not
delete results, overwrite markers, reuse a failed job ID, or resume under a
different source tree. Preserve failed namespaces and logs for audit.

After a code fix, use a new commit, a new job ID, and a new result
epoch/namespace when the old namespace contains incompatible or partial output.
Rerun fixture and scientific tests, then repeat the bounded-pilot gate.

Rollback source with a normal `git revert` of exact commits, never a destructive
reset. Code rollback must preserve durable results, manifests, completion
markers, provenance, and logs. Central registration rollback belongs only to a
ServerScheduler operator. Legacy scheduler deletion remains a separately
authorized migration action and is not implied by adapter success.

The project-side registration proposal and migration inventory are dated audit
snapshots, not live authority. Read the current central registration at
`/home/del6500/projects/ServerScheduler/config/projects/fedfisher.toml` before
making any registration claim, and do not write it without explicit permission.
