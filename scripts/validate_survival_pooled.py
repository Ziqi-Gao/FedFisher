#!/usr/bin/env python3
"""Validate completed pooled survival predictions without fitting or inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/del6500/projects/FedFisher')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    args = parser.parse_args(argv)
    from effect_modifier_experiment.survival_pooled.acceptance import (
        collect_acceptance, current_provenance, default_acceptance_path, _ordinary_path,
    )
    from effect_modifier_experiment.infrastructure.artifacts import create_bytes
    before = current_provenance()
    proof = collect_acceptance(args.inventory, args.cohort)
    output = default_acceptance_path(before, args.cohort)
    if current_provenance() != before:
        raise ValueError('source changed before pooled acceptance publication')
    _ordinary_path(output, exists=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    create_bytes(output, (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    if current_provenance() != before:
        raise ValueError('source changed while publishing pooled acceptance; preserve proof for review')
    print(json.dumps(dict(acceptance=str(output), validated_units=proof['validated_unit_count'],
                         submitted=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
