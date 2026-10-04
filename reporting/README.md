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
