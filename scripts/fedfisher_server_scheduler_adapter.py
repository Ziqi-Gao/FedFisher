#!/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python
"""Stable foreground entrypoint for reviewed FedFisher CPU shards.

Implementation: effect_modifier_experiment.scheduling.adapter. This path remains a compatible command entrypoint.
"""
from pathlib import Path
import sys

# Script execution does not otherwise put the repository on the import path.
if __name__ == '__main__':
    if Path(__file__).resolve() != Path('/home/del6500/projects/FedFisher/scripts/fedfisher_server_scheduler_adapter.py'):
        print('fedfisher_server_scheduler_adapter_error=entrypoint is outside canonical tree', file=sys.stderr)
        raise SystemExit(2)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_modifier_experiment.scheduling import adapter as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    # Preserve module identity for callers that import and configure the old API.
    sys.modules[__name__] = _implementation
