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

## Independent Ablation Contract

- The site-Gaussian ablation may change only the training covariate proposal
  distribution. Use the fixed 10-site mean-by-scale design documented under
  `site_gaussian_shift/`; preserve realized site sizes, outcome-label counts,
  the common control test set, and global intervention scope.
- The local-permutation ablation may change only prediction-intervention scope.
  Preserve the generated datasets and trained-model settings; split the common
  test set into disjoint site shards, permute only within a shard, and aggregate
  scalar feature scores with sample-size weights.
- Keep the two ablation variants mutually exclusive. A combined site-Gaussian
  plus local-permutation run is a separate factorial experiment and is not part
  of either approved ablation.
- Route every ablation through the existing
  `launch_synthetic_effect_modifier_300seeds_local.sh` -> slot -> worker
  scheduler chain. Select exactly one experiment with
  `EFFECT_MODIFIER_VARIANT=site_gaussian_shift` or
  `EFFECT_MODIFIER_VARIANT=local_permutation`. Do not add parallel GPU
  launch, slot, worker, or status scripts for these ablations.
- Durable outputs are strictly bound to
  `/data/del6500/FedFisher/effect_modifier_experiment/outputs/<RUN_ID>/`.
  Logs and runtime caches are strictly bound to the corresponding standard
  `/scr/del6500/FedFisher/effect_modifier_experiment/{logs,runtime}/<RUN_ID>/`
  trees. The per-GPU locks always use the single shared
  `/scr/del6500/FedFisher/effect_modifier_experiment/locks/` directory;
  `SCRATCH_ROOT` may not create a second lock namespace.
- RUN_ID namespaces are disjoint: control uses `prediction_intervention_*`,
  site-Gaussian uses `site_gaussian_shift_*`, and local permutation uses
  `local_permutation_*`. Workers must reject cross-variant names.
- Resolve and validate every scheduler write path before and after directory
  creation. `REPO_ROOT` must be this repository, durable output must remain
  under `/data/del6500/FedFisher`, and logs, locks, runtime files, and caches
  must remain under `/scr/del6500/FedFisher`. Reject symlink targets outside
  these roots.
- Every new output directory requires a format-v2 `run_manifest.env`
  containing commit, dirty-tree status, Git-status hash, source-tree hash
  including untracked files, the common run-configuration hash, and the explicit
  variant configuration. Refuse a nonempty directory without this manifest.
- Rerun the control under `prediction_intervention_ablation_control_300seeds`
  from the same finalized source tree as both ablations. The comparison tool
  must reject missing/legacy manifests and require identical provenance and
  common run-configuration hashes across all three experiments.
- Skip a completion marker only after its seed, RUN_ID, variant, provenance,
  common-configuration, and variant-configuration hashes match the run manifest.
  Status may report old
  markers separately as legacy/unvalidated but must not count them as validated.
- Ablation Screen sessions keep the existing exact prefix
  `fedfisher_em_<RUN_ID>_w` and the existing shared per-GPU lock files.
- Do not launch the full 300-seed jobs as part of code validation. Run the CPU
  regression test and shell syntax checks first, then obtain explicit launch
  confirmation.
