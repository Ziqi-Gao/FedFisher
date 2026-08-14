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
