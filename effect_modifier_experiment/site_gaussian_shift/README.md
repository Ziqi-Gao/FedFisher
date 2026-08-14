# Site-Gaussian Covariate-Shift Ablation

This is an independent one-change ablation of the control
`SyntheticEffectModifier` experiment. Training, models, algorithms, site
capacities, outcome-label counts, held-out test data, prediction intervention,
and seeds remain the same. The only changed input is the Gaussian covariate
distribution used to propose each site's training examples.

## Site distributions

For the default 10 sites, the fixed design is the Cartesian product

```text
site mean in {-0.75, -0.375, 0.0, 0.375, 0.75}
site std  in { 0.75,  1.25}
```

in mean-major order. At a site, all 100 raw covariates use the same scalar mean
and standard deviation. Each site uses an independent deterministic RNG stream.

The control site's realized sample size and two outcome-label counts are kept
exactly. To do that, candidate examples are generated from the site's Gaussian
DGP and retained by label until both control counts are reached. This conditions
the retained empirical sample on the requested labels; the output manifest
therefore records both the intended Gaussian parameters and realized raw-X
moments.

The global held-out test set remains the control `N(0, I)` test set and is
bitwise identical for a given seed. Prediction intervention remains global, so
this ablation does not also activate site-local reshuffling.

## Run

Do not combine this variant with
`--prediction_intervention_scope site` when running the intended one-change
ablation.

```bash
EFFECT_MODIFIER_VARIANT=site_gaussian_shift \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=site_gaussian_shift \
  bash scripts/status_synthetic_effect_modifier_local.sh
```

The existing shared launcher uses seeds 0 through 299, both MLPs, IID and
non-IID splits,
`fedavg`, `fedfisher_diag`, `fedfisher_kfac`, and the pooled prediction
baseline, matching the control run.

Default durable output:

```text
/data/del6500/FedFisher/effect_modifier_experiment/outputs/site_gaussian_shift_300seeds/
```

Each model/split run also writes
`*_site_gaussian_manifest.csv`, with site sizes, label counts, intended mean
and standard deviation, and empirical raw-X summaries.

Use `SEED_START` and `SEED_END` for a bounded smoke run. The shared
launcher waits for idle GPUs and is resumable through validated per-seed
completion markers. Its format-v2 `run_manifest.env` binds the output to RUN_ID,
source provenance, common run configuration, and the explicit variant settings.
The three comparison manifests must share provenance and common-config hashes. It must not
be launched on occupied GPUs without separate user authorization.
