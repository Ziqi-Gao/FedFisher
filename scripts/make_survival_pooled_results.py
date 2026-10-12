#!/usr/bin/env python3
"""Plot completed survival probability benchmarks without inference."""
import argparse
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from reporting.effect_modifier.survival_pooled import write_report
    write_report(args.acceptance, args.output)
    print(str(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
