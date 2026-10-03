"""Foreground reporting only; never starts a scientific workload."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

from effect_modifier_experiment.causal_comparison import protocol
from effect_modifier_experiment.causal_comparison.reporting import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("pilot", "formal"), required=True)
    parser.add_argument("--cells", nargs="+", choices=tuple(protocol.CELLS), required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.cells)) != len(args.cells):
        parser.error("cells must be unique")
    for path in write_report(protocol.PILOT_TASK if args.cohort == "pilot" else protocol.FORMAL_TASK,
                             args.cells, args.output_prefix):
        print(path)


if __name__ == "__main__":
    main()
