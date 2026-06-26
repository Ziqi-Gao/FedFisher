# Agent Guide

This directory documents the additive SyntheticEffectModifier experiment. It
does not replace the existing SyntheticBinary treatment/prediction-intervention
experiment.

## Scope

- Keep this experiment additive and isolated under `effect_modifier_experiment/`.
- Generated logs should go under `effect_modifier_experiment/logs/`.
- Generated outputs should go under `effect_modifier_experiment/outputs/`.
- Do not modify the original FedFisher aggregation path unless the user
  explicitly asks:
  - `run_one_shot_algs.py`
  - `algs/fisher_avg.py`
  - `train_model.py`
  - `utils/compress_fisher.py`

## Data Contract

- The dataset is `SyntheticEffectModifier`.
- It is a binary-outcome simulation with no raw `X` main effects and no
  treatment main effect.
- The model input is `[A, X1, ..., Xp, A*X1, ..., A*Xp]`.
- The outcome logit depends only on selected treatment-by-covariate interaction
  columns.
- Effect-modifier recovery must rank only the interaction block
  `A*X1, ..., A*Xp`.
- Do not report effect-modifier recovery from raw `X` columns alone.

## Defaults

- `effect_modifier_covariate_dim = 100`
- `effect_modifier_signal_dim = 10`
- `effect_modifier_signal_strength = 2.0`
- `effect_modifier_intercept = 0.0`
- `effect_modifier_treatment_prob = 0.5`
- With defaults, true interaction signal columns are zero-indexed `101..110`.

## Validation

- For changed Python files, run `python -m py_compile`.
- For Slurm scripts, run `bash -n`.
- Use `sbatch --test-only` when available before launching large arrays.
