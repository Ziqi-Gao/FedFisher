# FedFisher ServerScheduler migration inventory

Inventory timestamp: 2026-08-31 (America/Chicago). The initial inventory,
adapter, profiles, and fixture tests were followed by the project owner's
explicit selection of migration option 2: preserve pre-cutover artifacts but
abandon their continued recovery and scientific authority. No experiment was
launched, no result was changed, no legacy scheduler file was deleted, and no
ServerScheduler state or registration was written.

## Retained legacy scientific state

- The resumable main source tree is clean at
  `51c4e71d2ec35387fefaf4537fa4d8e7c27d0a55`.
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
running-job object plus `SERVER_SCHEDULER_*` allocation environment, dispatches
two fixed task names, and rejects caller-provided commands, paths, run IDs,
seeds, splits, models, and scientific hyperparameters.

Each admitted job is one independently resumable shard and one foreground
software worker. BLAS/OpenMP thread pools are fixed to the one allocated
physical core. The adapter calls the existing managers before and after every
unit of work, skips only validated markers, and refuses partial or invalid
output. Adapter caches live under
`/scr/del6500/FedFisher/effect_modifier_experiment/server_scheduler/JOB_ID`.
Scientific output remains under the existing durable output root.

Scheduler-era runs use fixed `server_scheduler_v1` run namespaces derived by
the adapter. Callers cannot select them. This prevents a future clean adapter
commit from colliding with or trying to reinterpret the preserved legacy
manifests, whose source provenance is intentionally different.

The dsLassoCov manager retains its legacy default. The adapter uses its new
`create_legacy_scheduler_dirs=False` option so central jobs do not create the
legacy local log or lock directories.

## Preliminary CPU task profiles

| Task | Shard parameters | CPU | Host memory minimum | Goal | GPU |
| --- | --- | --- | --- | --- | --- |
| `effect-modifier-dslassocov-shard` | variant; shard count 1, 2, or 4; valid index | fixed 1/1/1 core | 4,096 MiB | throughput | none |
| `effect-modifier-one-shot-lasso-shard` | variant; shard count 8 or 16; valid index | fixed 1/1/1 core | 8,192 MiB | throughput | none |

The memory floors are conservative derived profiles, not measured peak RSS:
the dsLassoCov path holds the 10-site synthetic design and sequential CV path;
the larger one-shot floor covers the deep MLP family and its sequential site
training. Historical logs contain no MaxRSS record. Registration must remain
disabled until an authorized pilot records actual peak RSS and runtime.

The GPU ablation task is intentionally absent. Existing code assumes eight
software threads, accepts a physical GPU index, overwrites
`CUDA_VISIBLE_DEVICES`, and is reached through idle polling and a project GPU
lock. No reviewed peak GPU-memory/utilization measurement exists. A future GPU
profile needs a separate adapter change and measurement/pilot stage.

## Registration proposal and examples

- Disabled proposal:
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

## Blockers before pilot or cutover

1. GPU resource requirements are unmeasured and the existing GPU worker has not
   yet been adapted to consume central visibility/device-zero semantics.
2. CPU memory values require an authorized measured pilot.
3. The central dispatcher ceiling of eight needs an explicit throughput-policy
   decision for 32/64 one-shot shards.
4. The dead Screen socket needs operator review/cleanup later; service and cron
   checks have the visibility limitations recorded above.
5. ServerScheduler registration remains the existing disabled, empty entry; no
   end-to-end pilot was authorized or performed.
6. The canonical repository's Git metadata is read-only in the current Codex
   sandbox, so the complete reviewed migration commit cannot be fetched or
   cherry-picked into the main tree from this session.

## Rollback

The complete migration implementation is isolated on branch
`codex/server-scheduler-adapter`, based on
`51c4e71d2ec35387fefaf4537fa4d8e7c27d0a55`, in the ignored staging clone
`/home/del6500/projects/FedFisher/results/server-scheduler-migration`.
Discarding that clone returns the main tree to the exact pre-migration state;
the main tree is already unchanged. The project rules, adapter, manager changes,
schemas, fixtures, proposal, and documentation are kept in one commit. If that
commit is later applied, rollback is a normal revert of the single commit. Data
and scratch output must not be deleted or rolled back.
