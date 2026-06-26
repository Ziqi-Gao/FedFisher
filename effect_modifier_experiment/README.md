# Synthetic Effect-Modifier Experiment

This experiment adds a clean interaction-only effect-modifier simulation to the
existing binary FedFisher pipeline. It preserves the original SyntheticBinary
experiment and the original FedFisher aggregation path.

## Data Model

For each example:

```text
X1, ..., Xp ~ N(0, 1)
A ~ Bernoulli(treatment_prob)
input = [A, X1, ..., Xp, A*X1, ..., A*Xp]
```

The binary outcome is generated from:

```text
logit P(Y = 1 | A, X) =
  intercept + sum_{j in S} gamma_j * A * Xj
```

There are no raw `X` main effects and no treatment main effect. The only
outcome-generating signal is in the treatment-by-covariate interaction block.
The default signal set is:

```text
S = {X1, ..., X10}
```

with alternating positive and negative interaction coefficients. With the
default `p=100`, input columns are zero-indexed as:

```text
0        = A
1..100   = X1..X100
101..200 = A*X1..A*X100
```

Therefore the true effect-modifier interaction columns are `101..110`. Recovery
metrics rank only the candidate interaction columns `101..200`; raw `X` columns
are not used to evaluate effect-modifier recovery.

## Federated Settings

The experiment reuses the existing settings:

- `iid`: split generated examples evenly across clients.
- `noniid`: split generated examples by outcome label using the existing
  class-wise Dirichlet allocation with `--alpha`.

The intended comparison is whether `fedfisher_diag` or `fedfisher_kfac` recover
the true interaction signals better than `fedavg`, especially under non-IID
client splits.

## Smoke Command

```bash
./.conda/bin/python main.py \
  --dataset SyntheticEffectModifier \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --effect_modifier_covariate_dim 100 \
  --effect_modifier_signal_dim 10 \
  --effect_modifier_signal_strength 2.0 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac \
  --prediction_intervention \
  --prediction_intervention_modes permute zero \
  --prediction_intervention_repeats 5 \
  --output_dir effect_modifier_experiment/outputs/prediction_intervention
```

Prediction-intervention detail and summary outputs include only interaction
candidate rows for this dataset. The standard one-shot accuracy CSV is still
written separately.

## Slurm Run

```bash
sbatch scripts/run_synthetic_effect_modifier_1000seeds.slurm
```

By default this runs `SyntheticMLP` and `SyntheticMLPDeep`, both `iid` and
`noniid`, 1000 seeds, and `fedavg`, `fedfisher_diag`, `fedfisher_kfac`.

Generated outputs and logs are ignored by Git under:

```text
effect_modifier_experiment/outputs/
effect_modifier_experiment/logs/
```
