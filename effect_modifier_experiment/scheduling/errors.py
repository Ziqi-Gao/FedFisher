"""Shared exception type across scheduler validation and foreground execution."""


class AdapterError(ValueError):
    """A fail-closed job-contract or task-contract violation."""
