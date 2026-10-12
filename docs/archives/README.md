# Historical result archive

The [recorded scheduler handoff](scheduler-handoff-20261012.md) is the complete,
unchanged earlier status record. Its historical observations and action lists
are not live authority. Current changing evidence remains in the
[canonical handoff](../codex/SCHEDULER_HANDOFF.md).

The active Git tree contains source, reviewed metadata and test fixtures.
Generated historical result tables are retained separately, indexed by
[historical-results.json](historical-results.json). This index lists every
original path, byte count and SHA-256 digest for 69 CSV files and their historical
comparison README. These are retained, non-authoritative results; the archive
does not change their scientific status or source provenance.

The hash-verified durable copy is stored at:

```text
/data/del6500/FedFisher/repository-archives/github-content-cleanup-20261012/
```

Original relative paths are preserved below that directory. Its
`archive-index.json` matches the checked-in index. Existing local originals
remain unchanged and ignored in the canonical checkout. A fresh clone omits
these generated tables; it does not automatically download the lab archive.

The files also remain available in Git history at commit
`a3784653f80b209ea3c7fe4cd11fd9efe94987c3`. For example, inspect an original table
without changing the current checkout:

```bash
git show a3784653f80b209ea3c7fe4cd11fd9efe94987c3:results/table2_summary.csv
```

Use explicit archived input paths for historical reporting. Report tables and
figures belong under `/data/del6500/FedFisher` or `/scr/del6500/FedFisher`.
The [binary comparison guide](../../reporting/README.md) documents the relevant
command. The preserved `results/server-scheduler-migration/` clone is unrelated
to this archive and must not be read, edited or executed for recovery.

Untracking retained files does not delete their bytes or rewrite Git history.
Do not remove, rename or overwrite the original historical artifacts, manifests
or proofs without the exact authorization required by the applicable project
policy.
