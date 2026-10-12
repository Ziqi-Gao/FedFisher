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

## Environments and dependencies

`requirements.txt` pins the scientific dependencies: NumPy, PyTorch and
NNGeometry. Preserve those pins during repository cleanup. `pyproject.toml`
holds optional static-tool settings; no new tool is required or installed.
The scientific adapter, storage paths and existing environment target the
canonical lab checkout. Cloning the repository elsewhere does not configure a
portable scientific runtime or register an execution path.

PNG reporting additionally uses Matplotlib, which is not part of the core
requirements. Effects replay selects `/opt/anaconda3/bin/python` for its
separate rendering step; the pooled-survival CLI imports Matplotlib in the
interpreter used to run that command. Treat these as lab-specific reporting
dependencies rather than replacements for the pinned science environment.
Rendering is limited to the deterministic completed-artifact exception. Figure
caches belong under project scratch, and tables and figures belong under the
data or scratch roots.

## Prepare a GitHub push

Complete the scientific closeout below before changing source bound to a
cohort. Review the Git diff and run focused checks appropriate to the changes.
Publish source, documentation, dependency specifications, small fixtures and
reviewed metadata. Keep datasets, fitted models, generated results, virtual
environments, caches, local credentials and machine-local settings out of new
commits. Existing historical artifacts, manifests and proofs remain intact;
repository cleanup does not authorize deleting them or rewriting their source
bindings. Do not traverse the preserved `results/server-scheduler-migration/`
clone to prepare a release.

Inspect changes before staging:

```bash
git status --short
git diff --stat
git diff --check
git diff
```

Stage explicit reviewed paths, then inspect the complete staged patch with
`git diff --cached --stat` and `git diff --cached`. Check newly added files for
generated artifacts, large binaries and credentials before committing. Ignore
rules affect untracked files; they do not remove an already tracked file from
the next commit. To retire tracked historical output from the active Git tree,
first preserve and hash-check a durable copy and record its recovery index.
Untrack exact reviewed paths with `git rm --cached -- PATH`, retaining the local
originals and Git history. Deleting or rewriting historical artifacts still
requires the separate authorization in the applicable policy. See the
[archive guide](archives/README.md). Confirm the intended branch and remote destination
before pushing; publishing Git history does not submit scheduler work.

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
