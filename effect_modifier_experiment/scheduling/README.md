# Project scheduling boundary

This package implements the project's side of ServerScheduler protocol v2.
Task routes and proposals do not establish live registration or admission.
The package cannot enable, register, centrally validate or submit a job.

| Module | Responsibility |
| --- | --- |
| `catalog.py` | Reviewed task names, fixed profile contracts and lazy module routes |
| `validation.py` | Strict running manifests, scientific parameters and allocation environment |
| `runtime.py` | Canonical-root guard, scratch/cache paths and allocated thread setup |
| `dispatch.py` | Selected runner loading after runtime setup |
| `adapter.py` | Stable manifest entrypoint API and CLI composition |
| `legacy.py` | Original CPU LASSO foreground orchestration |
| `errors.py` | Shared fail-closed adapter exception |
| `publication/` | Experiment plans, scientific gates, source review and immutable outbox composition |

The registered command remains
[`scripts/fedfisher_server_scheduler_adapter.py`](../../scripts/fedfisher_server_scheduler_adapter.py).
Protocols remain lightweight; numerical runners load after runtime setup.
Scientific task names, schemas, fixed defaults and output identities remain
owned by their existing contracts.

Pure request specifications live in each experiment's `requests.py` or
`recovery_requests.py`. Publication modules are `original`, `comparison`,
`effects_replay`, `fedfisher_timing`, `model_signal`, `model_signal_recovery`,
`survival` and `survival_recovery`. Each composes its existing admission checks
with shared immutable publication in [infrastructure](../infrastructure/README.md).
Old `scripts/prepare_*_requests.py` paths remain compatible commands.

Follow the [runbook](../../docs/codex/SCHEDULER_RUNBOOK.md) for new scientific
work. Schemas, disabled proposals, examples and audit records remain in
[server_scheduler](../server_scheduler/README.md). The
[architecture guide](../../docs/architecture.md) documents dependency rules.
