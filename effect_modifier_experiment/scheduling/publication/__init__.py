"""Immutable project outbox publication and experiment-specific admission gates.

Import named modules explicitly; package import performs no publication.
"""
from pathlib import Path


def require_canonical_source(module_file, project_root):
    """Reject code outside the source tree whose provenance would be published."""
    source = Path(module_file)
    expected = Path(project_root) / "effect_modifier_experiment/scheduling/publication" / source.name
    if source.is_symlink() or source.resolve(strict=True) != expected:
        raise ValueError("request publication requires the canonical registered source tree")
