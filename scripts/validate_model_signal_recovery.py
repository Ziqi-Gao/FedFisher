#!/usr/bin/env python3
"""Accept saved recovery pilot or formal evidence; never execute scientific work."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/home/del6500/projects/FedFisher')


def main(argv=None):
    from effect_modifier_experiment.model_signal_ablation import recovery_acceptance as gate

    return validate_saved_cohort(argv, gate)


def validate_saved_cohort(argv, gate):
    """The immutable proof publisher shared by separately reviewed epoch gates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--cohort', choices=('pilot', 'formal'), default='pilot')
    args = parser.parse_args(argv)
    provenance = gate.current_provenance()
    output = (gate.default_acceptance_path(provenance) if args.cohort == 'pilot' else
              gate.old._ordinary_path(gate.PROOF_ROOT / (
                  'formal-' + provenance['provenance_sha256'] + '.acceptance.json'), exists=False))
    pending = gate.old._ordinary_path(output.with_name('.' + output.name + '.pending'), exists=False)
    if pending.exists():
        raise FileExistsError('partial recovery proof must be preserved')
    prior = gate._binding(output) if output.exists() else None
    collect = gate.collect_pilot_acceptance if args.cohort == 'pilot' else gate.collect_formal_acceptance
    proof = collect(args.inventory)
    if proof['provenance'] != provenance or gate.current_provenance() != provenance:
        raise ValueError('source changed before recovery proof publication')
    payload = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if prior is not None:
        if gate._binding(output) != prior or gate.canonical(gate._read_json(output)) != gate.canonical(proof):
            raise ValueError('existing recovery proof differs from freshly validated evidence')
    else:
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
        if output.read_bytes() != payload:
            raise ValueError('recovery proof bytes changed after publication')
    if gate.current_provenance() != provenance:
        raise ValueError('source changed at final recovery proof readback')
    print(json.dumps(dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        published=prior is None, validated_requests=proof['validated_request_count'],
        validated_units=proof['validated_unit_count'], formal_eligible=proof.get('formal_eligible'),
        complete_units=proof.get('complete_units'), noncompleted_units=proof.get('noncompleted_units'),
        whole_cohort_scientific_success=proof.get('coverage', {}).get('whole_cohort_scientific_success'),
        submitted=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
