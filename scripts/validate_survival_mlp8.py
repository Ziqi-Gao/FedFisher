#!/usr/bin/env python3
"""Validate completed MLP8 survival artifacts without fitting or inference."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    args = parser.parse_args()
    from effect_modifier_experiment.survival_mlp8.acceptance import collect_acceptance, publish_acceptance
    proof = collect_acceptance(args.inventory, args.cohort)
    print(publish_acceptance(proof, args.cohort))
