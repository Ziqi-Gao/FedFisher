# Historical launch exports

`slurm/` preserves three historical Slurm scripts from the former `scripts/`
directory, with their original contents and experiment settings:

- `run_synthetic_effect_modifier_1000seeds.slurm`
- `run_synthetic_feature_importance_1000seeds.slurm`
- `run_synthetic_prediction_intervention_1000seeds.slurm`

They are retained for audit and are not authorized execution paths. New
computation follows the registered foreground ServerScheduler workflow in the
[runbook](../../docs/codex/SCHEDULER_RUNBOOK.md).

Their repository paths come from `REPO_ROOT`, Slurm's submission directory, or
the original cluster path; none depends on the script's current directory.
Historical resource requests and local output paths are preserved as written.
