# Development guide

Keep scientific execution on `/home/del6500/projects/FedFisher`. Read the root [AGENTS.md](../AGENTS.md)
and [project context](codex/PROJECT_CONTEXT.md) before changing source; read the
nearest experiment policy before touching that subtree. Runtime and scheduler
work also require the [runbook](codex/SCHEDULER_RUNBOOK.md) and applicable handoff.

## Add or change a module

Keep scientific configuration, data generation, estimators, evaluation scales
and semantic result validation in the owning experiment. Reuse shared timing,
fingerprinting and recovery mechanisms through `components/`. Put neutral
path/JSON/hash/provenance/publication/evidence operations in `infrastructure/`.
These infrastructure modules use the standard library and import no experiment.

Request plans belong to experiment `requests.py` or `recovery_requests.py`.
Compose publication and admission checks in `scheduling/publication/`. Resource
admission and physical allocation belong to ServerScheduler; project code has
no resource discovery, local queue, global concurrency or detached launch layer.
Keep protocols lightweight so manifest validation precedes numerical imports.

Use `computation.py`, `data_composition.py` or `prerequisites.py` when a library
needs scientific composition or prerequisite evidence. Importing another
experiment's runner or a `scripts` wrapper creates a reversed dependency.
Add a shared component only when the contracts actually agree. Preserve the
existing numerical order, RNG streams, recipes, tie handling and evaluation
scales during structural edits.

Put reusable reporting in `reporting/` and keep established command paths as
thin entrypoints. Preserve original algorithm interfaces and legacy audit
files as specified by the policies. See [architecture](architecture.md).

## Run development checks

Use the existing science interpreter; this repository layout requires no
package installation or dependency update:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py
```

For isolated development review, the fixture runner also accepts Git worktrees
below `/home/del6500/projects/FedFisher/.git/codex-review/`. It verifies the
worktree's Git metadata belongs below the canonical `.git/worktrees/` before
using that checkout for fixtures. Run the same command from the review tree.
The scientific scheduler adapter rejects a review tree: this fixture allowance
does not register an execution path, release frozen source or admit computation.

Select a focused family through the same cache/thread setup:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py --pattern 'test_survival_*.py'
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py --pattern 'test_repository_*.py'
```

The runner creates one foreground unittest process, disables bytecode, limits
numerical libraries to one thread and redirects temporary files and caches to
`/scr/del6500/FedFisher/tests/`. Tests also retain explicit project-scratch
fixture directories. Tests are categorized by their primary responsibility;
shared checkout paths live in `tests.paths`, and small fixture documents remain in
`tests/fixtures/`. See [the test guide](../tests/README.md).

`requirements.txt` remains the dependency specification. `pyproject.toml` holds
optional static-tool settings; no new tool is required or installed. The
original flat modules still require the canonical source import path, so this
layout does not advertise a standalone installed distribution.

## Scientific release boundary

Bounded tests verify numerical fixtures, contracts and import/CLI compatibility.
They do not authorize a scientific pilot or formal cohort. Before editing
source after a formal cohort, complete the runbook's source-bound scientific
closeout checks. Run new or materially changed task epochs only after schema,
profile, fixture, operator pilot and registration gates pass.

Never update an existing result epoch to substitute a new source hash. Partial
or invalid output fails closed without overwrite. A zero exit code does not
prove scientific completion. Record changing validation, proposal and blocker
evidence only in [SCHEDULER_HANDOFF.md](codex/SCHEDULER_HANDOFF.md); stable
architecture and procedure remain in the context and runbook.
