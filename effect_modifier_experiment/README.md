# Synthetic Effect-Modifier Experiment

This experiment tests whether one-shot federated models recover two treatment
effect modifiers in the presence of observed confounding, prognostic main
effects, unequal site sizes, and 98 noise covariates. It preserves the original
FedFisher aggregation path.

## Experiment navigation

This page defines the original binary DGP and factorial experiments. The
following additions keep separate scientific configurations and output identities:

| Area | Scope |
| --- | --- |
| [Original binary/base pipeline](../synthetic_binary_experiment/README.md) | Binary prediction and supervised signal recovery; root `data.py`, `models.py` and `main.py` |
| [causal_comparison](causal_comparison/README.md) | Multiround FedAvg, homogeneous R methods, common probability-scale evaluation and timing |
| [one_shot_effects](one_shot_effects/README.md) | FedFisher-EM, DR-Sieve, SCMV2 and common-protocol baselines |
| [one_shot_effects/diagnostics](one_shot_effects/diagnostics/README.md) | Saved-model gradient and approximation audits |
| [fedfisher_timing](fedfisher_timing/README.md) | Original Diag/KFAC fitting on CPU with separately measured timings |
| [survival](survival/README.md) | Site-local survival pseudo-observations, nonlinear dsLasso and neural working-Fisher adapters |
| [model_signal_ablation](model_signal_ablation/README.md) | Small-model/signal recipes and separately bound recovery |
| [components](components/README.md) | Shared measurement, fingerprints and tie-aware ranking arithmetic |
| [scheduling](scheduling/README.md) | Manifest validation, foreground dispatch and publication composition |
| [reporting](../reporting/README.md) | Completed-artifact tables and figures |
| [infrastructure](infrastructure/README.md) | Shared artifact, provenance, immutable request-publication and read-only execution-evidence helpers |
| [server_scheduler](server_scheduler/) | Scientific task schemas, registration proposals and request examples |
| [tests](../tests/) | Numerical, scientific-contract and scheduler fixtures |

Shared infrastructure is independent of these scientific recipes. It does not
replace an experiment's estimator, evaluation scale or acceptance conditions.

## Execution boundary

All new computation must be submitted through a centrally registered
ServerScheduler task and
`/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py`.
The adapter remains in the foreground, honors the allocated CPU set and thread
count, and resumes only through validated scientific completion markers.
It accepts only protocol-v2 running manifests and validates the selected
`cpu-1core` profile and complete scheduler environment. Protocol-v2
[request fixtures](../tests/fixtures/server_scheduler_requests/original-lasso/)
omit hardware fields for automatic profile selection.

The GNU Screen launchers, GPU polling, and local CPU/GPU lock scripts documented
below are legacy audit material and must not be used to submit new work.
Consult the [runbook](../docs/codex/SCHEDULER_RUNBOOK.md) for procedure and the
[handoff](../docs/codex/SCHEDULER_HANDOFF.md) for changing evidence; live central
registration determines admission. Every pre-cutover artifact is retained but
non-authoritative; see [legacy results policy](server_scheduler/legacy-results-policy.md).

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

## Historical direct smoke command (do not run on the current host)

This command documents the underlying scientific CLI only. It is not a
registered ServerScheduler task and is not an authorized current-host
execution path.

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

## Legacy 300-Seed GPU Run (do not submit)

The following section records the superseded pre-ServerScheduler workflow. It
is retained for provenance review and export only. The current host does not
authorize these commands for new work.

The legacy environment was prepared with:

```bash
bash scripts/setup_lab_server_env.sh
```

Legacy launch command (do not run):

```bash
bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
```

Each worker handles every fourth seed and runs both MLPs, IID and non-IID data,
and all three aggregators. If a GPU already has a compute process, its worker
waits until that GPU is empty for three consecutive one-minute polls. Completed
seeds receive durable markers and are skipped after a restart.

Legacy status commands:

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

## Factorial Ablations

The experiment now contains a complete 2x2 design over two switches:

| Variant | Site-Gaussian training proposal | Permutation scope |
| --- | --- | --- |
| `control` | off | global |
| `site_gaussian_shift` | on | global |
| `local_permutation` | off | site-local |
| `site_gaussian_local_permutation` | on | site-local |

The first two ablations remain one-change comparisons against control. The
[factorial combination](site_gaussian_local_permutation/README.md) directly
stacks the data shift from
[site-Gaussian covariate shift](site_gaussian_shift/README.md) with the
within-site reshuffling from
[site-local permutation](local_permutation/README.md). Seeds, models, existing
IID/non-IID splits, site capacities, algorithms, training hyperparameters, the
common held-out test tensors, and the pooled baseline remain fixed.

The following historical commands describe the pre-cutover factorial run and
must not be launched. A future rerun requires a separately registered GPU task
and approved pilot:

```bash
EFFECT_MODIFIER_VARIANT=control \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=site_gaussian_shift \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=local_permutation \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=site_gaussian_local_permutation \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
```

None of these launch commands is run automatically. After all results exist,
run:

```bash
python scripts/summarize_effect_modifier_ablations.py
```

The summary tool requires all four format-v2 manifests and identical
`provenance_sha256` and `run_config_sha256` values. It verifies all four switch
combinations, reports each arm versus control, and writes the per-seed factorial
interaction `combined - Gaussian - local + control`. Outputs include
`comparison_manifests.csv`, aggregate recovery and model summaries, paired
deltas, `factorial_interaction_seed_rows.csv`,
`factorial_interaction_summary.csv`, and local site summaries under
`/data/del6500/FedFisher/results/effect_modifier_ablations/` by default.

Because provenance includes the complete source tree, historical three-arm
outputs from an earlier commit cannot be mixed with a newly implemented
combined arm in the strict summary. All four cells must be rerun from the same
final commit.
