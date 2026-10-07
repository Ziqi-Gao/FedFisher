#!/usr/bin/env python3
"""Publish the fixed survival comparator cohort; submission belongs to the operator."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from effect_modifier_experiment.scheduling.publication.survival_comparators import main

if __name__ == '__main__':
    raise SystemExit(main())
