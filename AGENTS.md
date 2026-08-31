# FedFisher Agent Protocol

This file is the mandatory operating policy for every Codex session and every
agent working on the FedFisher project.

## Allowed write scope

Agents may create, edit, move, rename, or delete files only inside these three
directory trees:

1. `/home/del6500/projects/FedFisher`
2. `/data/del6500/FedFisher`
3. `/scr/del6500/FedFisher`

Everything outside these three trees is read-only. This restriction also covers
indirect writes such as package installation, generated caches, shell or editor
configuration, Git configuration, temporary files, permissions or ownership
changes, and writes performed through symlinks. Before writing through a
symlink, verify that its resolved target is inside one of the three allowed
trees.

If a task requires a write anywhere else, stop and obtain the user's explicit
permission for that exact location and action. Never broaden the write scope by
assumption.

## Directory responsibilities

### `/home/del6500/projects/FedFisher` — code and project metadata

Store the Git repository and small, human-maintained project files here,
including:

- Source code and tests
- Configuration files and small manifests
- Scripts, notebooks, and documentation
- Dependency lockfiles and environment specifications
- Git metadata and small sample or fixture data

Do not place large datasets, model weights, checkpoints, or disposable caches
in this directory.

### `/data/del6500/FedFisher` — persistent data and durable artifacts

Store large or long-lived files that must survive between runs here, including:

- Raw and processed datasets
- Trained model weights and important checkpoints
- Durable experiment results and exported artifacts
- Large files that are expensive or impossible to regenerate

Do not use this directory for transient caches, build scratch space, or routine
temporary files.

### `/scr/del6500/FedFisher` — scratch space and regenerable files

Store temporary, high-throughput, or regenerable files here, including:

- Download, dataset, and preprocessing caches
- Hugging Face, PyTorch, Triton, compiler, pip, and similar caches
- Temporary checkpoints and intermediate experiment outputs
- Runtime files, temporary logs, and build scratch space

Treat everything in this directory as disposable. Do not keep the only copy of
important code, data, model weights, or results here.

## Operational rules

- Keep source code in the project directory and reference data or scratch paths
  through configuration; do not duplicate the repository under `/data` or
  `/scr`.
- Redirect tool caches and temporary output into the FedFisher scratch
  directory instead of writing to `/tmp`, the user home directory, or global
  cache locations.
- Do not install or modify system packages, global environments, user-level
  environments, shell startup files, or applications outside the allowed
  trees.
- Reading files outside the allowed trees is permitted when needed for
  inspection or diagnosis, but do not expose credentials or unrelated private
  data.
- Destructive operations inside the allowed trees still require careful target
  verification and must remain within the user's requested scope.
- Report the location of newly generated code, data, models, and temporary
  outputs so the three-way separation remains auditable.

## ServerScheduler submission policy

This section governs all new computation and overrides legacy launch commands
that remain elsewhere in the repository for audit or rollback purposes.

- Submit every new compute task through the centrally registered
  ServerScheduler entrypoint. A program without a reviewed task name,
  parameter schema, resource profile, and fixture test is not authorized to
  run directly.
- The project may validate a scheduler-owned running-job manifest, but it must
  never write ServerScheduler pending, running, history, lease, audit, or log
  state. Scientific outputs continue to belong under the FedFisher `/data` and
  `/scr` roots.
- Resource selection belongs to ServerScheduler. Project code must not select
  physical CPUs or GPUs, implement global concurrency, poll devices, or create
  project resource locks for centrally managed jobs.
- The registered project entrypoint must stay in the foreground. It must not
  create Screen/tmux sessions, daemonize, detach, or background the scientific
  task.
- Use `SERVER_SCHEDULER_CPU_CORES`, the scheduler-provided CPU affinity, and
  one or fewer software workers/BLAS threads per allocated core. Every shard
  may process only its deterministic seed sequence.
- GPU tasks must consume the existing scheduler-provided
  `CUDA_VISIBLE_DEVICES` and treat the assigned device as CUDA device zero.
  FedFisher GPU task submission is currently disabled; legacy GPU launchers
  are not an authorized bypass.
- Delivery is at least once. A zero process exit status is not sufficient
  scientific completion evidence. Resume only by validating the project run
  manifest, provenance, semantic result identity, artifact hashes, and the
  task-specific completion marker. Partial or invalid output must fail closed
  without overwrite.
- The only initially registered FedFisher tasks are the CPU-only
  `effect-modifier-dslassocov-shard` and
  `effect-modifier-one-shot-lasso-shard`. All other workloads remain blocked
  until separately adapted, profiled, fixture-tested, piloted, and registered.

## Shared Compute Non-Interference

- Treat every process, Screen or tmux session, scheduler job, container, GPU
  context, and output tree not launched by the current FedFisher task as a
  foreign experiment, even when it uses a device needed by this project.
- Before launching GPU work, inspect device utilization, memory use, compute
  PIDs, process ownership, and existing FedFisher sessions. Report material
  contention to the user.
- Waiting for an idle GPU is the default. Running alongside an existing GPU
  experiment is an overlay launch and requires explicit user authorization for
  that launch. Authorization to overlay does not authorize any action against
  the existing experiment.
- Overlay launches must be additive and isolated: select devices with
  `CUDA_VISIBLE_DEVICES`, use only FedFisher-owned processes, sessions,
  environments, locks, logs, caches, runtime directories, and output paths,
  and preserve every pre-existing process and file unchanged.
- Never terminate, signal, pause, resume, renice, rebind CPU or GPU affinity,
  attach a debugger to, modify the environment or files of, or otherwise
  control a foreign experiment. Never use GPU reset, MIG reconfiguration,
  clock or power changes, CUDA MPS changes, driver operations, `killall`, or a
  broad `pkill` to obtain resources for FedFisher.
- If an overlay causes an out-of-memory error, resource conflict, instability,
  or unacceptable interference, stop or adjust only the exact FedFisher
  sessions and PIDs created by the current task. Resolve ownership and command
  lines with read-only checks before any stop action; leave all foreign work
  running and report the condition.
- Permission to start, restart, or stop FedFisher work never implies permission
  to interrupt another experiment. Any exception requires a separate, explicit
  user instruction naming the exact foreign target and action.

This policy may be changed only when the user explicitly requests a change to
the FedFisher agent protocol.
