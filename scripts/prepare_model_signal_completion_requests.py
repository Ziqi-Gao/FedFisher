#!/usr/bin/env python3
"""Compatibility command for reviewed post-refactor completion publication."""
import sys
sys.path.insert(0, '/home/del6500/projects/FedFisher')
from effect_modifier_experiment.scheduling.publication import model_signal_completion as _implementation
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
