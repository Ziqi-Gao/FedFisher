# Shared experiment infrastructure

This package contains neutral, standard-library helpers used by experiment
lifecycle code. It imports no scientific packages and calls no ServerScheduler
APIs. Scientific configurations, estimators, result validators, namespaces and
acceptance decisions remain in their experiment packages.

| Module | Responsibility |
| --- | --- |
| [artifacts.py](artifacts.py) | Allowed-root and symlink checks, strict/canonical JSON, hashes and immutable byte publication |
| [provenance.py](provenance.py) | Git/source provenance, canonical configuration hashing and environment-manifest encoding/reading |
| [requests.py](requests.py) | Immutable project outbox publication, with the complete inventory written last |
| [evidence.py](evidence.py) | Read-only central execution, producer-attempt, retry, log and audit evidence |

[Survival](../survival/README.md) and
[FedFisher timing](../fedfisher_timing/README.md) reuse these mechanisms while
retaining their own scientific contracts. Existing helper import paths retain
compatibility exports where needed; moving a helper does not merge experiment
formats or make historical results valid under a new source version.

The dependency direction is from scripts and experiment lifecycle modules into
this package, then into the Python standard library. It must not depend back on
an experiment's data, model, evaluation or protocol modules. Callers provide
the scientific identities and checks appropriate to their own recipes; there
is no universal experiment configuration here.

Outbox publication is not central submission. Execution-evidence readers do not
register tasks, dispatch jobs or mutate central state. Operational procedure is
defined by the [root protocol](../../AGENTS.md) and
[scheduler runbook](../../docs/codex/SCHEDULER_RUNBOOK.md). Return to the
[project navigation](../../README.md#project-navigation) or
[experiment index](../README.md#experiment-navigation) for scientific entry points.
