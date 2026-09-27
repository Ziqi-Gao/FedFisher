#!/usr/bin/env python3
"""Validate saved pilot evidence and publish one source-addressed proof."""
import argparse
import json
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/home/del6500/projects/FedFisher')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    args = parser.parse_args(argv)
    from effect_modifier_experiment.model_signal_ablation.acceptance import (
        _ordinary_path, collect_pilot_acceptance, current_provenance, default_acceptance_path)
    provenance = current_provenance()
    output = default_acceptance_path(provenance)
    pending = _ordinary_path(output.with_name('.' + output.name + '.pending'), exists=False)
    if output.exists() or pending.exists():
        raise FileExistsError('pilot proof or partial proof already exists; preserve without overwrite')
    proof = collect_pilot_acceptance(args.inventory)
    if proof['provenance'] != provenance or current_provenance() != provenance:
        raise ValueError('source changed before acceptance publication')
    output.parent.mkdir(parents=True, exist_ok=True)
    _ordinary_path(output, exists=False)
    _ordinary_path(pending, exists=False)
    encoded = (json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with pending.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if current_provenance() != provenance:
        raise ValueError('source changed while publishing proof; partial proof preserved')
    _ordinary_path(output, exists=False)
    os.link(pending, output)
    pending.unlink()
    if output.read_bytes() != encoded or current_provenance() != provenance:
        raise ValueError('published proof readback or source changed; preserve for review')
    print(json.dumps(dict(acceptance=str(output), validated_units=proof['validated_unit_count'],
                         submitted=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
