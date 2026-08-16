# Site-Gaussian plus Site-Local Permutation Factorial Arm

This experiment is the fourth cell of the `SyntheticEffectModifier` 2x2
factorial design. It directly combines the two previously independent
implementations:

- each site's training examples use the fixed site-specific Gaussian proposal;
- prediction-intervention permutation occurs only within that site's held-out
  shard.

No other setting changes. Seeds, realized training-site sizes, requested label
counts, outcome and treatment mechanisms, models, algorithms, training
hyperparameters, the common held-out test tensors, and pooled baselines match
the other factorial cells.

## Combined data and evaluation contract

For the default 10 sites, the training covariate proposals use the Cartesian
product

```text
site mean in {-0.75, -0.375, 0.0, 0.375, 0.75}
site std  in { 0.75,  1.25}
```

in mean-major order. All 100 raw covariates at one site use its scalar mean and
standard deviation. The control site's realized sample size and outcome-label
counts are preserved, and the common test tensors remain the control
`N(0, I)` test set.

The unchanged global test set is partitioned into 10 disjoint, exhaustive
shards using the same logic as the local-permutation arm. Each site receives
the trained global model and only its shard, permutes each raw feature within
that shard, rebuilds internally consistent treatment and control inputs, and
returns scalar feature scores. Scores are combined with sample-size weights
before Top-2 recovery and AUROC are computed.

Every model/split run writes both:

- `*_site_gaussian_manifest.csv`, containing intended and empirical site
  distribution information;
- `*_prediction_intervention_site_summary.csv`, containing site-level recovery
  metrics.

## Run

Use the existing shared GPU scheduler:

```bash
EFFECT_MODIFIER_VARIANT=site_gaussian_local_permutation \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
```

Check status with:

```bash
EFFECT_MODIFIER_VARIANT=site_gaussian_local_permutation \
  RUN_ID=site_gaussian_local_permutation_300seeds \
  bash scripts/status_synthetic_effect_modifier_local.sh
```

Default durable output:

```text
/data/del6500/FedFisher/effect_modifier_experiment/outputs/site_gaussian_local_permutation_300seeds/
```

Logs and runtime caches use the corresponding standard paths under
`/scr/del6500/FedFisher/effect_modifier_experiment/`; GPU locks remain in the
single shared lock directory. No additional launcher, slot, worker, status, or
GPU scheduling script is used.

Direct manual invocation of `main.py` with both low-level switches also
requires `--effect_modifier_combined_ablation`. The shared worker supplies this
opt-in automatically for this exact variant.

For a strict formal comparison, run control, the two one-change arms, and this
combined arm from the same finalized source tree. The summary requires all four
format-v2 manifests to share provenance and common configuration hashes, and
reports both combined-versus-control deltas and the factorial interaction
`combined - Gaussian - local + control`.
