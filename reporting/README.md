# Completed-artifact reporting

Report implementations are grouped by scientific area. Existing Python command
names and arguments remain available. New code imports package modules directly.

| Existing command in `scripts/` | Implementation |
| --- | --- |
| `make_synthetic_compare_visualizations.py` | [binary/comparison.py](binary/comparison.py) |
| `plot_synthetic_original.py` | [binary/original.py](binary/original.py) |
| `plot_synthetic_alpha_sweep.py` | [binary/alpha_sweep.py](binary/alpha_sweep.py) |
| `summarize_synthetic_original.py` | [binary/summary.py](binary/summary.py) |
| `make_completed_noniid_visualizations.py` | [effect_modifier/completed.py](effect_modifier/completed.py) |
| `make_completed_noniid_time_cost.py` | [effect_modifier/time_cost.py](effect_modifier/time_cost.py) |
| `make_effect_modifier_visualizations.py` | [effect_modifier/factorial.py](effect_modifier/factorial.py) |
| `make_one_shot_lasso_fedfisher_comparison.py` | [effect_modifier/lasso_fisher.py](effect_modifier/lasso_fisher.py) |
| `make_unified_noniid_comparison.py` | [effect_modifier/unified.py](effect_modifier/unified.py) |
| `report_causal_comparison.py` | [effect_modifier/causal_comparison.py](effect_modifier/causal_comparison.py) |
| `report_effects_replay.py` | [effect_modifier/effects_replay.py](effect_modifier/effects_replay.py) |
| `report_fedfisher_timing.py` | [effect_modifier/fedfisher_timing.py](effect_modifier/fedfisher_timing.py) |
| `summarize_effect_modifier_ablations.py` | [effect_modifier/ablations.py](effect_modifier/ablations.py) |
| `make_survival_pooled_results.py` | [effect_modifier/survival_pooled.py](effect_modifier/survival_pooled.py) |

These modules preserve their input contracts, metrics, aggregation and output
formats. Source-lineage hashes identify the actual implementation after a move.
Readers continue to enforce scientific artifact validation and their original
source requirements; reporting cannot convert incomplete computation into a
successful scientific observation.

Current-host local execution is limited to deterministic foreground reporting
of completed artifacts, with small presentation outputs below the FedFisher
data or scratch roots. It excludes data generation, fitting, model inference,
scientific completion mutation and detached work. Original binary historical
export paths remain references and do not grant a local-compute exception.
See the root [protocol](../AGENTS.md) and [architecture](../docs/architecture.md).

## Historical binary comparison inputs

`scripts/make_synthetic_compare_visualizations.py --help` lists explicit input
and output options. A fresh clone contains the renderer; its historical CSV
inputs are external artifacts. The default input archive is
`/data/del6500/FedFisher/repository-archives/github-content-cleanup-20261012`,
with the original relative paths preserved:

- `synthetic_binary_experiment/outputs/original_fedfisher_1000seeds/figures/original_compare_summary.csv`
- `synthetic_binary_experiment/outputs/original_fedfisher_alpha_sweep_1000seeds/figures/alpha_sweep_summary.csv`
- `results/compare/synthetic_prediction_model_summary.csv`
- `results/compare/synthetic_prediction_primary_metrics_summary.csv`
- `results/compare/synthetic_feature_importance_selected_summary.csv`

Set `--archive-root` to a copy with that layout, or use `--original-summary`,
`--alpha-summary` and `--summary-dir` to supply the saved tables explicitly.
The default uses the three already aggregated comparison tables. Explicit
`--prediction-dir` and `--feature-importance-dir` options instead aggregate
the corresponding saved per-seed CSVs with the original averaging rules;
missing raw inputs fail rather than falling back to another cohort.

The default output is `/data/del6500/FedFisher/results/compare`. `--output-dir`
must resolve below the FedFisher data or scratch roots, outside the preserved
repository archive, and must not overwrite an input table. Missing summaries
fail before output creation and identify the required path. Repository-local
historical outputs are never an implicit input or output destination. These
path options preserve the existing metrics and do not authorize new binary
training, evaluation or inference; the experiment's [policy](../synthetic_binary_experiment/AGENTS.md)
continues to apply.
