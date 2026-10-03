#!/usr/bin/env python3
"""Compatibility entrypoint for completed-artifact reporting.

Implementation: reporting.binary.summary. This path remains a compatible command entrypoint.
"""
from pathlib import Path
import sys

# Script execution does not otherwise put the repository on the import path.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reporting.binary import summary as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    # Preserve module identity for callers that import and configure the old API.
    sys.modules[__name__] = _implementation
