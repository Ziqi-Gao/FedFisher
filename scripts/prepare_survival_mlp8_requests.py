#!/usr/bin/env python3
"""Prepare a fixed MLP8 survival pilot for central operator submission."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from effect_modifier_experiment.scheduling.publication.survival_mlp8 import main

if __name__ == '__main__':
    raise SystemExit(main())
