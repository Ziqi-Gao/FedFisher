# Completed replay reporting

`scripts/report_effects_replay.py` is a deterministic foreground reporting
entrypoint. It requires all five fixed replay cells to pass
`ReplayArtifactStore.status()` under the current clean diagnostic source. It
checks original source bindings through that validator and reads the completed
JSON only; it never generates data, runs inference, fits a model, selects a
lambda, or changes scientific manifests or completion markers.

After scientific completion, use the unchanged diagnostic checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 \
/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python -B \
/home/del6500/projects/FedFisher/scripts/report_effects_replay.py \
--output-dir /data/del6500/FedFisher/effect_modifier_experiment/reports/effects-replay-v1-pilot-cohort-p1
```

The output must be a new directory below the FedFisher data or scratch root.
Symlinks, existing directories and partially generated reports are rejected.
`report.json` is written last and hashes the presentation files; it is not a
scientific completion marker. Choose a new report directory after a failed
presentation attempt rather than overwriting its evidence.

The scientific Python 3.9 environment performs all artifact validation. If
Matplotlib is absent there, one sequential, awaited foreground subprocess uses
the existing read-only `/opt/anaconda3/bin/python` presentation runtime
(Python 3.13, Matplotlib 3.10.6). That subprocess receives already-validated
JSON through stdin and imports only standard-library and plotting code. It
does not import the scientific artifact validator, models, or inference code.
No packages are installed or modified, no environment is copied, and nothing is
written below `/opt`. The actual renderer path and Matplotlib version are
recorded in each report. An unavailable plotting runtime fails before report
output creation; there is no recursive fallback or package auto-installation.

Before plotting imports, both parent and renderer redirect `MPLCONFIGDIR`,
`XDG_CACHE_HOME` and `TMPDIR` to
`/scr/del6500/FedFisher/effects-replay-report-cache`, disable Python bytecode,
select the Agg backend and limit numerical threads to one. The subprocess does
not detach or run parallel work. This is the project's existing completed-result
reporting exception, not a scientific scheduler bypass.

Outputs include Fisher path panels, SCMV2 CATE comparison bars, both saved DR
regularization paths, and CSVs containing prediction error moments, score
agreement, activation flips, model quality, T/V/H outcome cross-entropy and
Brier score, DR risk and group counts, coefficients and H rank intervals.
Prediction averaging and parameter FedAvg keep distinct labels. Original
V-selected lambda markers remain unchanged; test curves are diagnostic only.
Nonzero groups are not significant discoveries or FDR-controlled selections.

A failed or inconclusive gradient check produces a status row and omits the
uncomputed curves. An endpoint mismatch retains its computed path observations
but omits uncomputed model-quality figures; the status table records the
mismatch and the full-H gates. Legacy data have no CATE truth under this
protocol, so their CATE RMSE bar figures are omitted. No report authorizes
training, compression or new seeds.
