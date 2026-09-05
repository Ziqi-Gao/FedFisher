# FedFisher Agent Protocol

This file is the mandatory operating policy for every Codex session and every
agent working on the FedFisher project.

## Mandatory session bootstrap

Before modifying code, launching or requesting computation, or interpreting
scientific results:

1. Read `docs/codex/PROJECT_CONTEXT.md` completely.
2. For any computation request, outbox, scheduler, registration, submission,
   status, or runtime work, also read `docs/codex/SCHEDULER_RUNBOOK.md`
   completely.
3. For ServerScheduler integration or migration work, also read the current
   project handoff at `docs/codex/SCHEDULER_HANDOFF.md` completely.
4. When work touches an experiment subtree, read that subtree's nearest
   `AGENTS.md` even if the session started at the repository root.
5. Treat Git, scientific manifests, and ServerScheduler's live read-only state
   as authoritative for current status. Never infer live status from project
   documentation.
6. Use `/home/del6500/projects/FedFisher` as the only canonical source tree.
   `results/server-scheduler-migration/` is a preserved historical clone and
   must not be edited, executed, imported, or used as a Codex working
   directory.

The Codex context documents contain stable architecture and operating procedure
only. Put active migration status, tests, proposal/request evidence, transient
commit bindings, and blockers only in `docs/codex/SCHEDULER_HANDOFF.md`; do not
add those changing facts to the context or runbook.

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
- The foreground entrypoint accepts only protocol-v2 running manifests. It
  must validate `execution_profile`, the final allocation, the one-based
  attempt, the runtime estimate, and plural GPU identity fields against the
  complete scheduler-provided environment. Protocol-v1 singular GPU fields
  are invalid.
- Normal protocol-v2 requests omit `resources` and `execution_profile` so the
  scheduler can choose the registered profile. Project sessions may write a
  request only below `/scr/del6500/FedFisher/scheduler/outbox/`; a
  ServerScheduler operator validates and submits it. An outbox file is not a
  submitted job.
- Normal requests express scientific work only. They must not encode CPU/GPU
  counts, device identities, memory or utilization reservations, exclusivity,
  availability observations, or other machine choices in parameters, job IDs,
  configuration, or filenames.
- Resource selection belongs to ServerScheduler. Project code must not query
  `nvidia-smi` or another local resource ledger to decide whether or how to
  publish work; select physical CPUs or GPUs; implement global concurrency; or
  create project resource locks, leases, queues, Screen/tmux fan-out, or
  background schedulers for new jobs.
- The registered project entrypoint must stay in the foreground. It must not
  create Screen/tmux sessions, daemonize, detach, or background the scientific
  task.
- Use `SERVER_SCHEDULER_CPU_CORES`, the scheduler-provided CPU affinity, and
  one or fewer software workers/BLAS threads per allocated core. Every shard
  may process only its deterministic seed sequence.
- A GPU entrypoint must derive its fixed-per-attempt world size from both the
  running manifest and `SERVER_SCHEDULER_GPU_COUNT`, require them to agree with
  the plural GPU identity environment, preserve the scheduler-provided
  `CUDA_VISIBLE_DEVICES`, and use only logical CUDA ordinals `0..N-1`. It must
  never reinterpret those ordinals as host indices. A running attempt is never
  dynamically resized.
- A GPU profile may declare `kind = "gpu"` and
  `gpu_count_policy = "scheduler"` while omitting `gpu_count` only after tests
  prove that the same entrypoint and scientific result contract are valid for
  every centrally allowed count 1, 2, 3, and 4. Otherwise it must remain a
  truthful fixed-count profile. FedFisher has no registered GPU task now;
  legacy GPU launchers are not an authorized bypass.
- Project sessions may create only disabled registration proposals. They must
  not edit the central registration, enable a project or task, restart a
  scheduler service, or call central validate/submit/dispatch commands.
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
- New GPU work may start only after ServerScheduler admits it and supplies the
  allocation. A project session must not perform a direct or overlay launch,
  wait on a locally observed device, or alter visibility to choose host GPUs.
- Never terminate, signal, pause, resume, renice, rebind CPU or GPU affinity,
  attach a debugger to, modify the environment or files of, or otherwise
  control a foreign experiment. Never use GPU reset, MIG reconfiguration,
  clock or power changes, CUDA MPS changes, driver operations, `killall`, or a
  broad `pkill` to obtain resources for FedFisher.
- If a centrally admitted FedFisher task encounters an out-of-memory error,
  resource conflict, instability, or unacceptable interference, leave all
  foreign work unchanged and report the condition. Any action on the exact
  FedFisher attempt remains a separate, explicit operation within the
  scheduler ownership boundary.
- Permission to start, restart, or stop FedFisher work never implies permission
  to interrupt another experiment. Any exception requires a separate, explicit
  user instruction naming the exact foreign target and action.

This policy may be changed only when the user explicitly requests a change to
the FedFisher agent protocol.
