"""Compatibility alias for shared timing and payload observations."""

import sys
from ..components import measurement as _implementation

sys.modules[__name__] = _implementation
