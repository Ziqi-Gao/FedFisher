# Synthetic Effect-Modifier Experiment

This experiment tests whether one-shot federated models recover two treatment
effect modifiers in the presence of observed confounding, prognostic main
effects, unequal site sizes, and 98 noise covariates. It preserves the original
FedFisher aggregation path.

## Data Model

For each run, draw 10 site sizes independently:

```text
n_site ~ DiscreteUniform{500, ..., 1000}
```

For every example:

```text
X1, ..., X100 ~ N(0, 1)
input = [A, X1, ..., X100, A*X1, ..., A*X100]
```

Treatment assignment depends only on `X1` and `X2`:

```text
logit P(A = 1 | X) =
  logit(base_treatment_prob)
  + confounding_strength / sqrt(2) * (X1 + X2)
```

The resulting propensity is clipped to `[0.05, 0.95]` by default to preserve
overlap. The binary outcome is generated from:

```text
logit P(Y = 1 | A, X) =
  intercept
  + main_effect_strength / sqrt(2) * (X1 + X2)
  + signal_strength / sqrt(2) * (A*X1 - A*X2)
```

There is no treatment main effect. Therefore `X1` and `X2` are simultaneously:

- observed confounders, because they affect treatment assignment and outcome;
- prognostic variables, because they have first-order outcome effects;
- the only true effect modifiers, because only `A*X1` and `A*X2` enter the
  treatment interaction signal.

`X3..X100` and `A*X3..A*X100` are noise. With 100 covariates, zero-indexed input
columns are:

```text
0        = A
1..100   = X1..X100
101..200 = A*X1..A*X100
```

The true raw effect-modifier columns are `1, 2`, and their interaction columns
are `101, 102`.

## Federated Settings

- Default sites: `10`.
- Each site's training size is independently sampled from `500..1000`.
- The total training size therefore varies from `5000..10000` by seed.
- `iid`: shuffle the generated training population into the sampled site
  capacities.
- `noniid`: use capacity-constrained outcome-label Dirichlet assignment with
  `--alpha`, while exactly preserving every sampled site size.
- The held-out global test set remains independent and fixed at 10,000 examples
  by default.

The comparison remains `fedavg`, `fedfisher_diag`, and `fedfisher_kfac`, using
`SyntheticMLP` and `SyntheticMLPDeep`.

## Treatment-Contrast Evaluation

Primary recovery uses internally consistent counterfactual inputs:

```text
input_A1 = [1, X1, ..., X100, X1, ..., X100]
input_A0 = [0, X1, ..., X100, 0, ..., 0]
```

The model-estimated contrast is:

```text
tau_hat(X) = score(input_A1) - score(input_A0)
score = logit_Y1 - logit_Y0
```

Each raw covariate is permuted or zeroed, both counterfactual inputs are rebuilt,
and features are ranked by the mean absolute change in `tau_hat(X)`. The primary
metric is `tau_intervention_logit`; `tau_intervention_prob` is auxiliary.
Evaluation reports Top-2 hits/precision, signal ranks, and AUROC over all 100 raw
covariates. Direct interaction-column intervention is retained only as a
secondary diagnostic.

## Smoke Command

```bash
python main.py \
  --dataset SyntheticEffectModifier \
  --model SyntheticMLP \
  --num_clients 10 \
  --synthetic_split noniid \
  --alpha 0.1 \
  --effect_modifier_covariate_dim 100 \
  --effect_modifier_signal_dim 2 \
  --effect_modifier_signal_strength 2.0 \
  --effect_modifier_main_effect_strength 1.0 \
  --effect_modifier_confounding_strength 1.0 \
  --effect_modifier_propensity_clip 0.05 \
  --effect_modifier_site_min_samples 500 \
  --effect_modifier_site_max_samples 1000 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --prediction_intervention \
  --prediction_intervention_modes permute zero \
  --prediction_intervention_repeats 5 \
  --output_dir /data/del6500/FedFisher/effect_modifier_experiment/outputs/prediction_intervention
```

## 300-Seed Run on the Current Lab Server

The current host `fsmzhaolabgpu1.fsm.northwestern.edu` does not run Slurm. It
has four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, so the experiment
uses CUDA 12.8 PyTorch and detached GNU Screen workers instead of `sbatch`.
Prepare or repair the isolated environment with:

```bash
bash scripts/setup_lab_server_env.sh
```

Launch one resumable worker for each GPU:

```bash
bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
```

Each worker handles every fourth seed and runs both MLPs, IID and non-IID data,
and all three aggregators. If a GPU already has a compute process, its worker
waits until that GPU is empty for three consecutive one-minute polls. Completed
seeds receive durable markers and are skipped after a restart.

Check progress with:

```bash
bash scripts/status_synthetic_effect_modifier_local.sh
screen -ls
```

The default locations follow the repository storage protocol:

```text
/data/del6500/FedFisher/effect_modifier_experiment/outputs/prediction_intervention_ablation_control_300seeds/
    durable CSV outputs and completed-seed markers
/scr/del6500/FedFisher/envs/fedfisher-py39/
    regenerable Python environment
/scr/del6500/FedFisher/effect_modifier_experiment/logs/prediction_intervention_ablation_control_300seeds/
    local worker logs
/scr/del6500/FedFisher/effect_modifier_experiment/runtime/prediction_intervention_ablation_control_300seeds/
    caches and runtime files
```

Use `GPU_IDS`, `PYTHON_BIN`, or a namespace-compatible `RUN_ID` to select
execution details. `REPO_ROOT`, `DATA_ROOT`, and `SCRATCH_ROOT` must resolve to
the standard FedFisher roots. Durable output, logs, runtime files, and the
shared GPU lock directory are bound to their standard paths and cannot be
redirected through overrides or symlinks. `SEED_START` and `SEED_END` can bound
a test run; status reads that range from the manifest. The old `.slurm` runner
remains only for exporting the experiment to an actual Slurm cluster and is not
used on this server.

Every new run directory starts with a format-v2 `run_manifest.env`. It
records the Git commit, dirty-tree flag, Git-status hash, a source-tree hash
that includes untracked files, the common run-configuration hash, and an
explicit variant-configuration hash. A nonempty output directory without this
manifest is rejected. Completion markers are skipped only when RUN_ID, variant,
provenance, common configuration, and variant configuration all match. The
historical control output predates this format, so it is legacy/unvalidated and
cannot be used by the ablation summary.

## Independent Ablations

Two one-change ablations reuse the control's seeds, models, IID/non-IID splits,
site capacities, algorithms, training hyperparameters, and pooled baseline:

- [Site-Gaussian covariate shift](site_gaussian_shift/README.md) changes only
  the site-specific training covariate proposal distributions.
- [Site-local permutation](local_permutation/README.md) changes only the
  held-out reshuffling domain from global to within-site.

First rerun the control with the same finalized source tree, then launch the
two ablations without changing the code between launches:

```bash
EFFECT_MODIFIER_VARIANT=control \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=site_gaussian_shift \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=local_permutation \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
```

None of these launch commands is run automatically. After results exist, compare both
ablations against the control with:

```bash
python scripts/summarize_effect_modifier_ablations.py
```

The summary tool requires all three format-v2 manifests, rejects the legacy
control, and requires identical `provenance_sha256` and `run_config_sha256`
values across the three runs. It also verifies each variant's two allowed
difference flags. It writes `comparison_manifests.csv`, aggregate recovery,
model diagnostics, paired per-seed deltas, paired delta summaries, and local
site summaries under
`/data/del6500/FedFisher/results/effect_modifier_ablations/` by default.
