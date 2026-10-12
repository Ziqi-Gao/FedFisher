# Development validation

Run from the canonical checkout or a verified isolated review worktree with
the existing science interpreter:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py
```

The runner starts one foreground unittest process, disables bytecode, limits
numerical libraries to one thread and places temporary files/caches under
`/scr/del6500/FedFisher/tests/`. Existing tests may also use explicit
project-scratch fixture directories. No environment is installed.

Review fixtures are limited to Git worktrees below the canonical
`.git/codex-review/` directory whose metadata resolves below `.git/worktrees/`.
The runner uses that checkout's modules. The scientific scheduler adapter
rejects review worktrees; passing fixtures does not release frozen canonical
source or authorize a scientific pilot/formal job.

| Directory | Primary responsibility |
| --- | --- |
| `unit/` | Small data, estimator, evaluation and shared mechanism fixtures |
| `contracts/` | Fixed protocols, semantic identities, manifests and immutable artifacts |
| `integration/` | Adapter/runner, request, acceptance and reporting composition |
| `regression/` | Known stationarity, separation, stopping and gradient failure cases |
| `architecture/` | Dependency boundaries, compatible commands and exact request parity |
| `fixtures/server_scheduler/` | Scheduler-owned running-manifest examples for adapter fixtures |
| `fixtures/server_scheduler_requests/` | Unsubmitted request examples grouped by recipe; see the [contract index](../effect_modifier_experiment/server_scheduler/README.md) |
| Other `fixtures/` files | Small reference data, including scientific request-parity digests |

Mixed historical files stay intact and are placed by primary responsibility;
their numerical tests and assertions remain together. `tests.paths.PROJECT_ROOT`
provides the shared checkout location. Cross-test helpers use explicit
package imports. The runner discovers with `-s tests -t .` to retain one module
identity. Plain `unittest discover -s tests` remains recursive, but callers must
still provide the required project-scratch cache and one-thread environment.

Focused checks use the same setup:

```bash
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py --pattern 'test_repository_*.py'
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B scripts/run_tests.py --pattern 'test_survival_*.py'
```

Scheduler fixture manifests are test inputs, not submissions. Small numerical
tests do not authorize scientific pilots or formal jobs. Golden request digests
compare scientific parameters, configurations and seed coverage independently
of file placement; review-source hashes change when implementations move.
See [development](../docs/development.md) for scientific release gates.
