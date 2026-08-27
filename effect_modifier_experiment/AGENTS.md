# Agent Guide

This directory documents the additive `SyntheticEffectModifier` experiment. It
does not replace the existing `SyntheticBinary` prediction-intervention
experiment.

## Scope

- Keep this experiment additive and isolated under `effect_modifier_experiment/`.
- Store durable generated outputs under
  `/data/del6500/FedFisher/effect_modifier_experiment/outputs/`.
- Store local worker logs, runtime files, locks, and caches under
  `/scr/del6500/FedFisher/effect_modifier_experiment/`.
- Keep source, small documentation, and job definitions in the repository.
- Do not modify the original FedFisher aggregation path unless the user
  explicitly asks:
  - `run_one_shot_algs.py`
  - `algs/fisher_avg.py`
  - `train_model.py`
  - `utils/compress_fisher.py`

## Data Contract

- The dataset is `SyntheticEffectModifier`.
- Use `10` sites by default.
- Draw each site's training size independently and uniformly from the integers
  `500..1000`; do not force equal site sizes.
- Use `100` raw Gaussian covariates.
- Only `X1` and `X2` may affect treatment assignment or outcome generation.
  `X3..X100` and their treatment interactions are noise.
- The model input is `[A, X1, ..., X100, A*X1, ..., A*X100]`.
- Treatment assignment must depend on `X1` and `X2`, with clipped propensities
  to preserve overlap. Thus `X1` and `X2` are genuine observed confounders.
- The outcome logit contains first-order effects for `X1` and `X2`, plus
  treatment interactions `A*X1` and `A*X2`. There is no treatment main effect.
- `X1` and `X2` are the only true effect modifiers.
- Primary recovery must rank raw `X1..X100` by how much perturbing each `Xi`
  changes the model-estimated treatment contrast
  `tau_hat(X) = score(A=1, X) - score(A=0, X)`.
- Counterfactual inputs must remain internally consistent: `[1, X, X]` for
  treatment and `[0, X, 0]` for control.
- Report Top-2 recovery and AUROC over all 100 raw covariates.
- Direct intervention on `A*X1..A*X100` is a secondary diagnostic only.

## Defaults

- `num_clients = 10` for this dataset
- `effect_modifier_site_min_samples = 500`
- `effect_modifier_site_max_samples = 1000`
- `effect_modifier_covariate_dim = 100`
- `effect_modifier_signal_dim = 2`
- `effect_modifier_signal_strength = 2.0`
- `effect_modifier_main_effect_strength = 1.0`
- `effect_modifier_confounding_strength = 1.0`
- `effect_modifier_intercept = 0.0`
- `effect_modifier_treatment_prob = 0.5` at `X1=X2=0`
- `effect_modifier_propensity_clip = 0.05`
- True raw effect-modifier columns are zero-indexed `1..2`; their interaction
  columns are zero-indexed `101..102`.

## Current Execution Environment

- The active target is the lab server `fsmzhaolabgpu1.fsm.northwestern.edu`.
- This host has four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs and does
  not run Slurm. Do not use `sbatch` for this experiment on this host.
- Use `scripts/setup_lab_server_env.sh` to maintain the isolated environment at
  `/scr/del6500/FedFisher/envs/fedfisher-py39`.
- Blackwell `sm_120` requires the CUDA 12.8 PyTorch build pinned in
  `requirements.txt`; the old PyTorch 2.1/CUDA 11.8 Quest environment is not
  valid here.
- Launch the 300-seed experiment with
  `scripts/launch_synthetic_effect_modifier_300seeds_local.sh`. It creates one
  detached GNU Screen worker per selected GPU and resumes from per-seed done
  markers.
- By default each slot must wait until its physical GPU has no compute process
  for three consecutive polls. Do not bypass that guard while another user's
  process owns the GPU unless the user explicitly authorizes sharing.
- The `.slurm` runner is retained only as a legacy/export path for a real Slurm
  cluster; it is not the current-server entry point.

## Shared-GPU Overlay Procedure

- Normal launches keep `WAIT_FOR_GPU_IDLE=1`; each slot waits for three
  consecutive idle polls before starting its worker.
- Only after the user explicitly authorizes sharing already occupied GPUs may
  an agent launch this experiment with:

  ```bash
  WAIT_FOR_GPU_IDLE=0 bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
  ```

- Before an overlay launch, record the existing GPU compute PIDs, owners,
  utilization, and memory use. Verify that the FedFisher allocation has
  sufficient memory headroom, and tell the user that compute throughput may be
  shared.
- Overlay workers must remain one independently visible process per selected
  physical GPU, selected through `CUDA_VISIBLE_DEVICES`, and must write only to
  the FedFisher `/data` and `/scr` trees defined above.
- The only sessions that this experiment may control are those it created with
  the exact prefix `fedfisher_em_<RUN_ID>_w`. Before stopping or restarting a
  session, resolve its exact Screen name and child command line.
- Never kill, suspend, renice, reset, or otherwise manipulate a pre-existing
  process to make room. In particular, do not use `killall`, broad `pkill`,
  `nvidia-smi --gpu-reset`, MIG changes, clock or power changes, CUDA MPS
  changes, or edits to another experiment's code, environment, data, logs,
  checkpoints, or outputs.
- If coexistence fails, terminate only the exact FedFisher Screen sessions or
  FedFisher child PIDs from the current run. Other experiments must continue
  unchanged, regardless of their GPU utilization or runtime.

## Validation

- For changed Python files, run a syntax check without writing caches into the
  repository, followed by a small CPU data-contract check when dependencies are
  available.
- Run `bash -n` on every changed shell or Slurm script.
- Validate a one-element CUDA kernel and both NNGeometry `PMatDiag` and
  `PMatKFAC` before launching workers after an environment change.

## Factorial Ablation Contract

- The site-Gaussian one-change ablation may change only the training covariate
  proposal distribution. Use the fixed 10-site mean-by-scale design documented
  under `site_gaussian_shift/`; preserve realized site sizes, outcome-label
  counts, the common control test set, and global intervention scope.
- The local-permutation one-change ablation may change only
  prediction-intervention scope. Preserve the generated datasets and
  trained-model settings; split the common test set into disjoint site shards,
  permute only within a shard, and aggregate scalar feature scores with
  sample-size weights.
- The approved factorial arm is `site_gaussian_local_permutation`: enable the
  same site-Gaussian training proposal as the first arm and the same site-local
  intervention scope as the second arm, without changing any other setting.
  Direct CLI use of both low-level switches requires the explicit
  `--effect_modifier_combined_ablation` opt-in.
- Route every arm through the existing
  `launch_synthetic_effect_modifier_300seeds_local.sh` -> slot -> worker
  scheduler chain. Select exactly one experiment with
  `EFFECT_MODIFIER_VARIANT=control`, `site_gaussian_shift`,
  `local_permutation`, or `site_gaussian_local_permutation`. Do not add parallel
  GPU launch, slot, worker, or status scripts for these experiments.
- Durable outputs are strictly bound to
  `/data/del6500/FedFisher/effect_modifier_experiment/outputs/<RUN_ID>/`.
  Logs and runtime caches are strictly bound to the corresponding standard
  `/scr/del6500/FedFisher/effect_modifier_experiment/{logs,runtime}/<RUN_ID>/`
  trees. The per-GPU locks always use the single shared
  `/scr/del6500/FedFisher/effect_modifier_experiment/locks/` directory;
  `SCRATCH_ROOT` may not create a second lock namespace.
- RUN_ID namespaces are disjoint: control uses `prediction_intervention_*`,
  site-Gaussian uses `site_gaussian_shift_*`, local permutation uses
  `local_permutation_*`, and the factorial arm uses
  `site_gaussian_local_permutation_*`. Workers must reject cross-variant names.
- Resolve and validate every scheduler write path before and after directory
  creation. `REPO_ROOT` must be this repository, durable output must remain
  under `/data/del6500/FedFisher`, and logs, locks, runtime files, and caches
  must remain under `/scr/del6500/FedFisher`. Reject symlink targets outside
  these roots.
- Every new output directory requires a format-v2 `run_manifest.env`
  containing commit, dirty-tree status, Git-status hash, source-tree hash
  including untracked files, the common run-configuration hash, and the
  explicit variant configuration. Refuse a nonempty directory without this
  manifest.
- Run control and all three factorial arms from the same finalized source tree.
  The comparison tool must reject missing/legacy manifests and require
  identical provenance and common run-configuration hashes across all four
  experiments. It must report each arm versus control and the per-seed
  difference-in-differences interaction `combined - Gaussian - local + control`.
- Skip a completion marker only after its seed, RUN_ID, variant, provenance,
  common-configuration, and variant-configuration hashes match the run
  manifest. Status may report old markers separately as legacy/unvalidated but
  must not count them as validated.
- Ablation Screen sessions keep the existing exact prefix
  `fedfisher_em_<RUN_ID>_w` and the existing shared per-GPU lock files.
- Do not launch full 300-seed jobs as part of code validation. Run the CPU
  regression test, shell syntax checks, and a bounded combined smoke run first,
  then obtain explicit launch confirmation.

## dsLassoCov-style CPU Benchmark Contract

- The additive benchmark name is **dsLassoCov-style federated logistic LASSO
  (Python benchmark; not official R/DataSHIELD dsLassoCov)**. It is a
  single-process cross-silo simulation of site loss/gradient aggregation, not
  a deployment privacy or DataSHIELD disclosure-control boundary.
- Reuse the existing `SyntheticEffectModifier` generator, split, site-Gaussian
  proposal, common test set, and local-shard evaluator exactly. The primary
  endpoint remains raw-covariate tau-intervention Top-2/AUROC; absolute
  interaction-coefficient recovery is auxiliary only.
- This benchmark is CPU-only. Its launcher and workers must not inspect or set
  GPU visibility, acquire the existing per-GPU locks, enter the GPU scheduling
  chain, wait for a GPU, or modify any GPU state.
- Durable results are restricted to
  `/data/del6500/FedFisher/effect_modifier_experiment/outputs/<RUN_ID>/`.
  Logs, runtime files, and caches are restricted to the matching
  `/scr/del6500/FedFisher/effect_modifier_experiment/{logs,runtime}/<RUN_ID>/`
  trees. CPU locks alone use the separate, exact
  `/scr/del6500/FedFisher/effect_modifier_experiment/cpu_locks/<RUN_ID>/`
  namespace; they must never use or alter the GPU `locks/` namespace.
- RUN_IDs must use the `dslassocov_<variant>_` namespace and all output,
  scratch, and lock paths must reject symlinks. CPU scheduling defaults to two
  workers with one thread each. Both the shell and Python front doors must
  enforce `CPU_WORKERS * THREADS_PER_WORKER <= 4`.
- A nonblocking launch lock serializes launcher setup for a RUN_ID. A
  nonblocking per-seed lock is held continuously across IID fit, non-IID fit,
  validation, and done-marker creation, and is released on validated skip,
  success, error, or signal. Detached Screen descendants must not inherit the
  launch-lock descriptor. Lock files record auditable owner/PID events.
- The CPU scripts may control only Screen sessions with their exact
  `fedfisher_em_dslassocov_<RUN_ID>_w` prefix and locks below their exact
  RUN_ID directory. Never kill, suspend, renice, reorder, or otherwise modify
  J-lens, an existing ablation, or any other process or experiment.
- Format-v2 manifests must record and canonical-hash the run, benchmark/CV/
  optimizer, and variant configurations plus commit, dirty/status, and source
  provenance. Every manager, worker, status, resume, and invariant entry point
  must recompute those hashes and the current source provenance. Partial
  artifacts are never completion evidence; only a fully validated done marker
  may be resumed.
- Control/local-permutation and site-Gaussian/combined must preserve matching
  training, test, and fit fingerprints. Validate those pairs across a
  same-source four-cell seed-0 smoke.
- Before any full 300-seed launch, run no-cache Python syntax checks, shell
  syntax checks, all regression tests (including real lock contention and FD
  lifetime), the four-cell bounded smoke, and a default 5-fold/30-lambda
  full-dimension single-seed/single-split timing pilot, then stop for review.
