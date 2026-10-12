# Command entrypoints

Scientific execution uses only the compatible foreground
[`fedfisher_server_scheduler_adapter.py`](fedfisher_server_scheduler_adapter.py)
with a centrally admitted protocol-v2 running manifest. Request commands publish
review bundles; they do not submit jobs. Import package implementations in new
library code.

`scripts` is an explicit Python package. Module-forwarding commands, including
`scripts.report_effects_replay` and the `prepare_*` commands, share their package
implementation's state on import. Some commands instead parse arguments and
call a package function: `make_survival_pooled_results.py` delegates to
`write_report`. Established filenames remain CLI entrypoints. New libraries use
the implementation packages directly.

| Compatible command | Implementation |
| --- | --- |
| `fedfisher_server_scheduler_adapter.py` | [scheduling/adapter.py](../effect_modifier_experiment/scheduling/adapter.py) |
| `prepare_fedfisher_scheduler_requests.py` | [publication/original.py](../effect_modifier_experiment/scheduling/publication/original.py) |
| `prepare_comparison_pilot_requests.py` | [publication/comparison.py](../effect_modifier_experiment/scheduling/publication/comparison.py) |
| `prepare_effects_replay_requests.py` | [publication/effects_replay.py](../effect_modifier_experiment/scheduling/publication/effects_replay.py) |
| `prepare_fedfisher_timing_requests.py` | [publication/fedfisher_timing.py](../effect_modifier_experiment/scheduling/publication/fedfisher_timing.py) |
| `prepare_model_signal_requests.py` | [publication/model_signal.py](../effect_modifier_experiment/scheduling/publication/model_signal.py) |
| `prepare_model_signal_recovery_requests.py` | [publication/model_signal_recovery.py](../effect_modifier_experiment/scheduling/publication/model_signal_recovery.py) |
| `prepare_model_signal_completion_requests.py` | [publication/model_signal_completion.py](../effect_modifier_experiment/scheduling/publication/model_signal_completion.py) |
| `prepare_survival_requests.py` | [publication/survival.py](../effect_modifier_experiment/scheduling/publication/survival.py) |
| `prepare_survival_recovery_requests.py` | [publication/survival_recovery.py](../effect_modifier_experiment/scheduling/publication/survival_recovery.py) |
| `prepare_survival_pooled_requests.py` | [publication/survival_pooled.py](../effect_modifier_experiment/scheduling/publication/survival_pooled.py) |
| `prepare_survival_mlp8_requests.py` | [publication/survival_mlp8.py](../effect_modifier_experiment/scheduling/publication/survival_mlp8.py) |
| `prepare_survival_comparators_requests.py` | [publication/survival_comparators.py](../effect_modifier_experiment/scheduling/publication/survival_comparators.py) |
| `make_synthetic_compare_visualizations.py` | [reporting/binary/comparison.py](../reporting/binary/comparison.py) |
| `plot_synthetic_original.py` | [reporting/binary/original.py](../reporting/binary/original.py) |
| `plot_synthetic_alpha_sweep.py` | [reporting/binary/alpha_sweep.py](../reporting/binary/alpha_sweep.py) |
| `summarize_synthetic_original.py` | [reporting/binary/summary.py](../reporting/binary/summary.py) |
| `make_completed_noniid_visualizations.py` | [reporting/effect_modifier/completed.py](../reporting/effect_modifier/completed.py) |
| `make_completed_noniid_time_cost.py` | [reporting/effect_modifier/time_cost.py](../reporting/effect_modifier/time_cost.py) |
| `make_effect_modifier_visualizations.py` | [reporting/effect_modifier/factorial.py](../reporting/effect_modifier/factorial.py) |
| `make_one_shot_lasso_fedfisher_comparison.py` | [reporting/effect_modifier/lasso_fisher.py](../reporting/effect_modifier/lasso_fisher.py) |
| `make_unified_noniid_comparison.py` | [reporting/effect_modifier/unified.py](../reporting/effect_modifier/unified.py) |
| `report_causal_comparison.py` | [reporting/effect_modifier/causal_comparison.py](../reporting/effect_modifier/causal_comparison.py) |
| `report_effects_replay.py` | [reporting/effect_modifier/effects_replay.py](../reporting/effect_modifier/effects_replay.py) |
| `report_fedfisher_timing.py` | [reporting/effect_modifier/fedfisher_timing.py](../reporting/effect_modifier/fedfisher_timing.py) |
| `summarize_effect_modifier_ablations.py` | [reporting/effect_modifier/ablations.py](../reporting/effect_modifier/ablations.py) |
| `make_survival_pooled_results.py` | [reporting/effect_modifier/survival_pooled.py](../reporting/effect_modifier/survival_pooled.py), via `write_report` |
| `check_effect_modifier_dslassocov_invariants.py` | [effect_modifier_experiment/dslassocov/invariants.py](../effect_modifier_experiment/dslassocov/invariants.py) |
| `check_effect_modifier_one_shot_lasso_invariants.py` | [effect_modifier_experiment/one_shot_lasso/invariants.py](../effect_modifier_experiment/one_shot_lasso/invariants.py) |

`validate_*` and `check_*_invariants.py` remain scientific acceptance/checking
commands with their existing provenance and completion requirements.
`validate_model_signal_completion.py`, `validate_survival_pooled.py`,
`validate_survival_mlp8.py` and `validate_survival_comparators.py` cover their
separate experiment contracts. The MLP8 preparation and validation commands use
`--discovery` for the complete V2 probability and Top-2/AUROC comparison; omitting
it retains the historical V1 prediction contract. Refer to each experiment's
guide before preparing requests or validating an artifact namespace.

`run_tests.py` runs bounded development fixtures. `setup_lab_server_env.sh`
remains the existing reviewed environment setup; a layout change needs no
installation. See [development](../docs/development.md).

Historical `launch_*.sh`, `run_*_worker.sh`, `run_*_slot.sh`, `status_*.sh`,
`*_cpu_lock.sh` and `.slurm` files retain their original locations for audit or
export. Their Screen, resource-polling and lock workflows do not authorize new
local experiments. Follow the [runbook](../docs/codex/SCHEDULER_RUNBOOK.md) and
refer to the [complete earlier README](../docs/legacy/root-readme.md).
