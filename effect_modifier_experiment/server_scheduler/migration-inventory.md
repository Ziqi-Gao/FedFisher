# FedFisher ServerScheduler migration inventory

This file is a historical migration record. For the current project handoff,
including live-status snapshots, latest tests, proposal/request evidence, and
blockers, read `docs/codex/SCHEDULER_HANDOFF.md` and verify its claims against
Git and ServerScheduler's read-only durable state.

Inventory first recorded: 2026-08-31; updated after the first central CPU pilot
on 2026-09-01 (America/Chicago). The project owner selected migration option 2:
preserve pre-cutover artifacts but abandon their continued recovery and
scientific authority. Migration commits `1b43497` and `d05073f` established the
protocol-v2 adapter, after which an authorized ServerScheduler operator enabled
the CPU-only registration and submitted one bounded dsLassoCov pilot. That
pilot failed closed as documented below. No legacy scheduler file or scientific
artifact was deleted or overwritten, and GPU submission remains blocked.

## Retained legacy scientific state

- The legacy resumable source tree was clean at
  `51c4e71d2ec35387fefaf4537fa4d8e7c27d0a55`. The canonical project tree is
  based on protocol-v2 baseline
  `d05073febb0277a419867cbe1f9d7ce68644b396`. A repeat pilot must use a clean
  commit containing the first-pilot fix described below.
- Formal one-shot manifests are bound to that clean source tree. The observed
  aggregate was 2,960 validated cells of 7,200, with 4,240 missing and no known
  invalid/partial cells. The owner explicitly abandoned further recovery so a
  single canonical scheduler-era source tree can replace that provenance.
- All paths below
  `/data/del6500/FedFisher/effect_modifier_experiment/outputs` are retained.
  Run manifests, result metadata, artifact hashes, completion markers, and
  logs are not deletion candidates, but every pre-cutover artifact is now
  non-authoritative and must not be resumed or mixed with scheduler-era output.
  See `legacy-results-policy.md`.
- `scripts/run_synthetic_effect_modifier_local_worker.sh`,
  `scripts/run_effect_modifier_dslassocov_cpu_worker.sh`, and
  `scripts/run_effect_modifier_one_shot_lasso_cpu_worker.sh` remain preserved.
  The new adapter calls the same Python managers and scientific runners directly
  in the foreground; it does not replace the scientific algorithms.

## Local scheduler surfaces

| Workload | Resource-selection caller | Resource wrapper/state | Scientific/resume owner | Status surface |
| --- | --- | --- | --- | --- |
| Four GPU ablations | `scripts/launch_synthetic_effect_modifier_300seeds_local.sh` | `scripts/run_synthetic_effect_modifier_local_slot.sh`, GNU Screen, `nvidia-smi`, per-GPU `flock` files | `scripts/run_synthetic_effect_modifier_local_worker.sh`, run manifests, seed markers | `scripts/status_synthetic_effect_modifier_local.sh` |
| CPU dsLassoCov | `scripts/launch_effect_modifier_dslassocov_cpu.sh` | GNU Screen and `scripts/effect_modifier_dslassocov_cpu_lock.sh` | `scripts/run_effect_modifier_dslassocov_cpu_worker.sh`, `manage_dslassocov_cpu_run.py`, seed markers | `scripts/status_effect_modifier_dslassocov_cpu.sh` |
| CPU formal one-shot LASSO | `scripts/launch_effect_modifier_one_shot_lasso_cpu.sh` | GNU Screen and `scripts/effect_modifier_one_shot_cpu_lock.sh` | `scripts/run_effect_modifier_one_shot_lasso_cpu_worker.sh`, `manage_one_shot_lasso_cpu_run.py`, cell markers | `scripts/status_effect_modifier_one_shot_lasso_cpu.sh` |

Scheduler references also remain in:

- `effect_modifier_experiment/AGENTS.md`
- `effect_modifier_experiment/README.md`
- `effect_modifier_experiment/dslassocov/README.md`
- `effect_modifier_experiment/one_shot_lasso/AGENTS.md`
- `effect_modifier_experiment/one_shot_lasso/README.md`
- `effect_modifier_experiment/dslassocov_run_contract.py`
- `tests/test_effect_modifier_dslassocov.py`
- `tests/test_effect_modifier_one_shot_cpu_scheduler.py`

These references are retained until an approved pilot, central cutover, and
separate legacy-deletion review. They are explicitly marked as legacy in the
project rules and relevant READMEs.

## Runtime inventory

- FedFisher worker/manager/runner processes: none visible in the current process
  namespace. Exact worker-name probes returned no process; broad matches were
  only the inspection command itself.
- Screen: direct `screen -ls` cannot operate because `/run/screen` is mode 0775,
  but the socket tree contains exactly
  `/run/screen/S-del6500/3772541.fedfisher_em_dslassocov_dslassocov_site_gaussian_local_permutation_300seeds_w`.
  The migration instructions identify it as dead. It was not wiped.
- Held FedFisher file locks: none reported by `lslocks`.
- Persistent legacy lock files: 4 files under
  `/scr/del6500/FedFisher/effect_modifier_experiment/locks`, 1,213 files under
  `/scr/del6500/FedFisher/effect_modifier_experiment/cpu_locks`, and 74 files
  under `/scr/del6500/FedFisher/effect_modifier_experiment/one_shot_cpu_locks`.
  File presence is not treated as proof of a live holder.
- ServerScheduler leases/state referring to FedFisher: none found under the
  readable `/data/del6500/ServerScheduler` and `/scr/del6500/ServerScheduler`
  trees. No state directory was created.
- Services: no matching FedFisher unit file was found in the readable system or
  user unit directories. The user D-Bus was unavailable, so live user-unit
  enumeration could not be independently confirmed. `crontab -l` was denied by
  PAM, and `/var/spool/cron` was unreadable.

## Adapter ownership boundary

The proposed stable executable is
`/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py`.
It accepts only `--job-manifest RUNNING_JOB_JSON`, validates the complete
protocol-v2 running-job object, selected execution profile, optional request
resources, final allocation, attempt, runtime estimate, and plural-GPU
`SERVER_SCHEDULER_*` environment, dispatches two fixed task names, and rejects
caller-provided commands, paths, run IDs, seeds, splits, models, and scientific
hyperparameters.

Each admitted job is one independently resumable shard and one foreground
software worker. BLAS/OpenMP thread pools are fixed to the one allocated
physical core. The adapter calls the existing managers before and after every
unit of work, skips only validated markers, and refuses partial or invalid
output. Adapter caches live under
`/scr/del6500/FedFisher/effect_modifier_experiment/server_scheduler/JOB_ID`.
Scientific output remains under the existing durable output root.

Scheduler-era runs use fixed `server_scheduler_v2` run namespaces derived by
the adapter. Callers cannot select them. This prevents a future clean adapter
commit from colliding with or trying to reinterpret the preserved legacy
manifests, whose source provenance is intentionally different.

The dsLassoCov manager retains its legacy default. The adapter uses its new
`create_legacy_scheduler_dirs=False` option so central jobs do not create the
legacy local log or lock directories.

## Protocol-v2 CPU task profiles

| Task | Profile | Shard parameters | CPU | Host memory minimum | Initial runtime | Goal | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `effect-modifier-dslassocov-shard` | `cpu-1core` | variant; shard count 1, 2, or 4; valid index | fixed 1/1/1 core | 4,096 MiB | 36,000 s | throughput | none |
| `effect-modifier-one-shot-lasso-shard` | `cpu-1core` | variant; shard count 8 or 16; valid index | fixed 1/1/1 core | 8,192 MiB | 86,400 s | throughput | none |

The memory floors are conservative derived profiles, not measured peak RSS:
the dsLassoCov path holds the 10-site synthetic design and sequential CV path;
the larger one-shot floor covers the deep MLP family and its sequential site
training. Historical logs contain no MaxRSS record. The dsLassoCov runtime is a
10-hour conservative ceiling over complete legacy one-core runs whose observed
maximum was about 7 hours 48 minutes. No complete comparable one-shot shard was
recorded, so its 24-hour value is a conservative scheduling placeholder. Both
fixed profiles use `cpu_scaling_efficiency = 0.0`. Registration must remain
disabled until an authorized pilot records actual peak RSS and calibrates
runtime.

The GPU ablation task is intentionally absent. Existing code assumes eight
software threads, accepts a physical GPU index, overwrites
`CUDA_VISIBLE_DEVICES`, and is reached through idle polling and a project GPU
lock. No reviewed peak GPU-memory/utilization measurement exists. A future GPU
profile needs a separate adapter change and measurement/pilot stage.

## Registration proposal and examples

- Disabled protocol-v2 proposal:
  `effect_modifier_experiment/server_scheduler/registration-proposal.toml`
- dsLassoCov parameter schema:
  `effect_modifier_experiment/server_scheduler/dslassocov-shard-parameters.schema.json`
- One-shot parameter schema:
  `effect_modifier_experiment/server_scheduler/one-shot-lasso-shard-parameters.schema.json`
- Example requests:
  `effect_modifier_experiment/server_scheduler/examples/dslassocov-shard.job-request.json`
  and
  `effect_modifier_experiment/server_scheduler/examples/one-shot-lasso-shard.job-request.json`

The installed ServerScheduler currently has `max_parallel_jobs = 8`. It cannot
recreate the legacy 32/64-way simultaneous one-shot layout without an operator
policy change; the same shards could run in admitted waves. That policy decision
is outside this migration stage.

## Protocol-v2 validation and first pilot

The protocol-v2 follow-up was verified without launching scientific work:

- 16 FedFisher adapter fixture, allowlist, environment, retry, marker, and path
  tests passed.
- 5 one-shot CPU configuration, shard, shell-safety, and lock tests passed.
- 18 selected dsLassoCov scientific, thread-reuse, manifest, atomic-write,
  completion-marker, and CPU/GPU-boundary tests passed. Legacy Screen launch
  tests were not run.
- The current ServerScheduler v2 loader accepted the project proposal, both
  request examples, both running-job fixtures, and the then-disabled central
  FedFisher registration. The operator later enabled it for the CPU pilot.
- The project proposal and installed central FedFisher registration were byte
  identical at validation time.
- ServerScheduler `doctor` parsed all registrations and reported 64 effective
  physical CPU cores, but remained nonhealthy because the host `nvidia-smi`
  query could not communicate with the driver and observed 0 of 4 configured
  GPUs. FedFisher GPU policy remains disabled.

The first central pilot used job
`fedfisher-dslassocov-control-pilot-d05073f-shard-0-of-4`, profile
`cpu-1core`, CPU ID 0 on NUMA node 0, and 4096 MiB. Three attempts started.
Attempt 1 wrote only the seed-0 IID artifacts, then failed before non-IID when
the same process called `torch.set_num_interop_threads(1)` a second time after
parallel work had started. Attempts 2 and 3 correctly detected the partial
seed and refused overwrite; the job reached `failed` with no validated seed
marker. The manifest remained provenance-valid for `d05073f`.

The failed output namespace
`dslassocov_control_server_scheduler_v2_300seeds` and all three scheduler log
pairs are retained as audit evidence. The fix makes thread configuration
idempotent only when the already configured inter-op count is exactly one and
moves all four dsLassoCov variants to the disjoint
`dslassocov_<variant>_server_scheduler_v2r1_300seeds` result epoch. A different
existing inter-op count still fails closed. No repeat pilot has run yet.

## Exact later deletion candidates

No item below is authorized for deletion now. After an approved pilot, drain,
cutover, zero-holder check, and documentation update, review exactly:

Source candidates:

1. `/home/del6500/projects/FedFisher/scripts/launch_synthetic_effect_modifier_300seeds_local.sh`
2. `/home/del6500/projects/FedFisher/scripts/run_synthetic_effect_modifier_local_slot.sh`
3. `/home/del6500/projects/FedFisher/scripts/launch_effect_modifier_dslassocov_cpu.sh`
4. `/home/del6500/projects/FedFisher/scripts/effect_modifier_dslassocov_cpu_lock.sh`
5. `/home/del6500/projects/FedFisher/scripts/launch_effect_modifier_one_shot_lasso_cpu.sh`
6. `/home/del6500/projects/FedFisher/scripts/effect_modifier_one_shot_cpu_lock.sh`
7. `/home/del6500/projects/FedFisher/scripts/status_synthetic_effect_modifier_local.sh`
8. `/home/del6500/projects/FedFisher/scripts/status_effect_modifier_dslassocov_cpu.sh`
9. `/home/del6500/projects/FedFisher/scripts/status_effect_modifier_one_shot_lasso_cpu.sh`

Scratch candidates:

1. `/scr/del6500/FedFisher/effect_modifier_experiment/locks`
2. `/scr/del6500/FedFisher/effect_modifier_experiment/cpu_locks`
3. `/scr/del6500/FedFisher/effect_modifier_experiment/one_shot_cpu_locks`

External operator-only candidate (outside FedFisher write scope):

1. `/run/screen/S-del6500/3772541.fedfisher_em_dslassocov_dslassocov_site_gaussian_local_permutation_300seeds_w`

The three scientific worker scripts, all `/data` results, manifests, provenance
records, completion markers, Python scientific/status managers, and historical
logs are explicitly not deletion candidates.

## Blockers before a successful pilot or cutover

1. The thread-initialization fix and v2r1 namespace change must pass regression
   tests, be committed, and be used by a new clean-provenance pilot request.
2. CPU memory and initial runtime values still require a successful measured
   pilot before production enablement.
3. GPU resource requirements are unmeasured and the existing GPU worker has not
   yet been adapted to consume central visibility/device-zero semantics. This
   does not block the registered CPU pilot, but GPU submission remains blocked.
4. The central dispatcher ceiling of eight needs an explicit throughput-policy
   decision for 32/64 one-shot shards.
5. The dead Screen socket needs operator review/cleanup later; service and cron
   checks have the visibility limitations recorded above.
6. The registration is CPU-pilot-enabled, but legacy callers have not been cut
   over or deleted and formal production activation has not been accepted.
7. ServerScheduler's global `doctor` currently fails because the host NVIDIA
   driver is unavailable (`nvidia-smi` exit 9; 0 of 4 configured GPUs
   observed). FedFisher's registered tasks are CPU-only, but proceeding with a
   pilot while the global control plane is nonhealthy requires an explicit
   ServerScheduler operator decision.

## Rollback

The last clean pre-migration project commit is
`51c4e71d2ec35387fefaf4537fa4d8e7c27d0a55`. The applied migration commits are
`1b43497e80ed843f2726593504a4cdca0497f5fa` and
`d05073febb0277a419867cbe1f9d7ce68644b396`; the committed first-pilot fix will
extend that rollback boundary. Rollback must use normal Git revert operations
on exact commits. Data, manifests, completion markers, logs, and scratch output
must not be deleted or rolled back. The ServerScheduler repository currently
has no commit history, so its
central registration still lacks an independent Git rollback point and must be
committed by the ServerScheduler operator before production activation.
