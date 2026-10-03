"""Bind publication fixtures to the checkout under test, never production source."""
from importlib import import_module
from pathlib import Path
from unittest import mock

from tests.paths import PROJECT_ROOT


def bind_review_sources(stack, publisher):
    """Patch both owners of metadata while retaining real source hashing checks.

    Production plans deliberately bind the canonical registered source tree.
    An isolated development checkout may exercise publication only with mocked
    provenance and scratch outputs; its source metadata must describe that same
    checkout rather than borrowing the frozen canonical implementation.
    """
    plan = import_module(publisher.build_requests.__module__)
    if PROJECT_ROOT not in Path(plan.__file__).resolve().parents:
        raise AssertionError("request plan was imported from a different checkout")
    original_root = plan.PROJECT_ROOT
    sources = {
        name: PROJECT_ROOT / path.relative_to(original_root)
        for name, path in plan.REVIEW_SOURCES.items()
    }
    for module in (plan, publisher):
        stack.enter_context(mock.patch.object(module, "PROJECT_ROOT", PROJECT_ROOT))
        stack.enter_context(mock.patch.object(module, "REVIEW_SOURCES", sources))
    return plan
