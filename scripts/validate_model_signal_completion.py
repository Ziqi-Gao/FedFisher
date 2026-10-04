#!/usr/bin/env python3
"""Validate saved post-refactor completion evidence without scientific execution."""
import sys
sys.path.insert(0, '/home/del6500/projects/FedFisher')


def main(argv=None):
    from effect_modifier_experiment.model_signal_ablation import completion_acceptance as gate
    from scripts.validate_model_signal_recovery import validate_saved_cohort
    return validate_saved_cohort(argv, gate)


if __name__ == '__main__':
    raise SystemExit(main())
