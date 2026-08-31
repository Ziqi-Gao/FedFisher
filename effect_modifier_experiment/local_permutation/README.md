# Site-Local Permutation Ablation

This is an independent one-change ablation of the control
`SyntheticEffectModifier` experiment. The generated training and test tensors,
training procedure, models, algorithms, seeds, and intervention definitions are
unchanged. Only the domain over which each feature is reshuffled changes from
the entire held-out test set to each site's own held-out shard.

GPU execution for this ablation is not registered with ServerScheduler and is
therefore blocked. The commands below document the retained pre-cutover
workflow only; they must not launch new work. Existing output is preserved but
non-authoritative under `../server_scheduler/legacy-results-policy.md`.

## Local evaluation contract

For each seed, the unchanged global held-out test set is partitioned into 10
disjoint shards:

- shard capacities are proportional to the realized training-site sizes;
- IID runs use a shuffled capacity split;
- non-IID runs use capacity-constrained outcome-label assignment with the same
  Dirichlet alpha;
- every test example belongs to exactly one shard.

Each site receives the trained global model and only its local shard. For every
raw covariate, the site permutes values within that shard, rebuilds internally
consistent `[1, X, X]` and `[0, X, 0]` inputs, and measures the change in
`tau_hat(X)`. Raw site scores are never pooled before intervention. Aggregate
feature scores are sample-size weighted across sites, then Top-2 recovery and
AUROC are computed from those aggregate scores.

The same procedure is used for the secondary direct-interaction diagnostic.
Site-level Top-2/AUROC rows are written separately. Zero intervention is also
computed locally; because it does not depend on a permutation domain, its
sample-size-weighted scores should match global zero on the same test union up
to floating-point reduction error.

## Legacy run record (do not launch)

This one-change variant does not enable the site-Gaussian shift. Use the
[combined factorial arm](../site_gaussian_local_permutation/README.md) when both
switches must be enabled.

```bash
EFFECT_MODIFIER_VARIANT=local_permutation \
  bash scripts/launch_synthetic_effect_modifier_300seeds_local.sh
EFFECT_MODIFIER_VARIANT=local_permutation \
  bash scripts/status_synthetic_effect_modifier_local.sh
```

The legacy shared launcher used seeds 0 through 299, both MLPs, IID and
non-IID splits,
`fedavg`, `fedfisher_diag`, `fedfisher_kfac`, and the pooled prediction
baseline, matching the control run.

Default durable output:

```text
/data/del6500/FedFisher/effect_modifier_experiment/outputs/local_permutation_300seeds/
```

Aggregate results are in the normal prediction-intervention CSVs.
`*_prediction_intervention_site_summary.csv` contains per-site recovery
metrics and `permutation_scope=site` identifies the ablation in every output
table.

Historically, `SEED_START` and `SEED_END` bounded a smoke run. The shared
launcher waited for idle GPUs and resumed through validated per-seed
completion markers. Its format-v2 `run_manifest.env` binds the output to RUN_ID,
source provenance, common run configuration, and the explicit variant settings.
All four factorial manifests had to share provenance and common-config hashes.
A future rerun requires a reviewed central GPU task and approved pilot.
