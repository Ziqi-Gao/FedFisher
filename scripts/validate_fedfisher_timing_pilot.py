#!/usr/bin/env python3
"""Read-only scientific pilot checks; save a new project-side acceptance proof."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/del6500/projects/FedFisher')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    from effect_modifier_experiment.causal_comparison.artifacts import safe_path, create_bytes
    from effect_modifier_experiment.fedfisher_timing.acceptance import collect_pilot_acceptance
    output = safe_path(args.output)
    if output.exists():
        raise FileExistsError('acceptance proof already exists')
    proof = collect_pilot_acceptance(args.inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_path(output)
    create_bytes(output, (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())
    print(json.dumps(dict(acceptance=str(output), validated_units=proof['validated_unit_count'],
                         submitted=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
