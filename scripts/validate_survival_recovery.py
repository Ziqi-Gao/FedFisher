#!/usr/bin/env python3
"""Accept saved recovery diagnostics or the mixed-version formal comparison."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/home/del6500/projects/FedFisher')


def main(argv=None):
    from effect_modifier_experiment.survival import recovery_acceptance as gate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--cohort', choices=('diagnostic', 'formal'), default='diagnostic')
    args = parser.parse_args(argv)
    provenance = gate.current_provenance()
    output = gate.default_acceptance_path(provenance, args.cohort)
    pending = gate.old._ordinary_path(output.with_name('.' + output.name + '.pending'), exists=False)
    if output.exists() or pending.exists():
        raise FileExistsError('existing or partial recovery proof must be preserved')
    collect = gate.collect_diagnostic_acceptance if args.cohort == 'diagnostic' else gate.collect_formal_acceptance
    proof = collect(args.inventory)
    if proof['provenance'] != provenance or gate.current_provenance() != provenance:
        raise ValueError('source changed before recovery proof publication')
    payload = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    gate.old._ordinary_path(output, exists=False)
    gate.old._ordinary_path(pending, exists=False)
    with pending.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if gate.current_provenance() != provenance:
        raise ValueError('source changed while publishing; partial proof preserved')
    os.link(pending, output)
    pending.unlink()
    if output.read_bytes() != payload or gate.current_provenance() != provenance:
        raise ValueError('proof readback or final source check failed; preserve evidence')
    print(json.dumps(dict(path=str(output), validated_units=proof['validated_unit_count'],
        formal_eligible=proof.get('formal_eligible'), selected_recipe=proof.get('selected_recipe'),
        submitted=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
