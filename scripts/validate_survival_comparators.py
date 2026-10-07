#!/usr/bin/env python3
"""Validate saved survival comparator artifacts without model inference."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from effect_modifier_experiment.survival_comparators.acceptance import main

if __name__ == '__main__':
    raise SystemExit(main())
