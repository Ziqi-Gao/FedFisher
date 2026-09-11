"""Read completed cells, verify paired invariants, and aggregate small tables."""

import csv
import io
import math
from pathlib import Path
from statistics import mean, stdev

from . import protocol
from .artifacts import (ArtifactStore, OUTPUT_ROOT, canonical, current_provenance,
                        make_manifest, read_json, safe_path)
from ..dslassocov_run_contract import atomic_create_text


def metrics(result):
    recovery = result["evaluation"]["metrics"]["permute/tau_prob"]["recovery"]
    stages = result["timing"]["stages"]
    return dict(expected_topk_hits=recovery["expected_topk_hits"],
                expected_topk_precision=recovery["expected_topk_precision"],
                probability_all_signals_selected=recovery["probability_all_signals_selected"],
                auroc=recovery["auroc"],
                fit_wall_seconds=sum(stages[key]["wall_seconds"] for key in ("train", "tuning", "aggregation")),
                fit_cpu_seconds=sum(stages[key]["cpu_seconds"] for key in ("train", "tuning", "aggregation")),
                evaluation_wall_seconds=stages["evaluation"]["wall_seconds"])


def validate_pairs(records):
    """Input contains exactly one complete four-variant cell/seed/split block."""
    if set(records) != set(protocol.VARIANTS):
        raise ValueError("all four variants are required for a paired comparison")
    if len({r["fingerprints"]["test"] for r in records.values()}) != 1:
        raise ValueError("common test fingerprint differs across variants")
    for first, second in (("control", "local_permutation"),
                          ("site_gaussian_shift", "site_gaussian_local_permutation")):
        a, b = records[first], records[second]
        for key in ("training", "model"):
            if a["fingerprints"][key] != b["fingerprints"][key]:
                raise ValueError("paired " + key + " fingerprints differ")
        if a["fit"]["fit_fingerprint"] != b["fit"]["fit_fingerprint"]:
            raise ValueError("paired fit fingerprints differ")
    for first, second in (("control", "site_gaussian_shift"),
                          ("local_permutation", "site_gaussian_local_permutation")):
        if records[first]["evaluation"]["permutation_sha256"] != records[second]["evaluation"]["permutation_sha256"]:
            raise ValueError("matched intervention scopes used different permutations")


def aggregate(rows):
    buckets = {}
    for row in rows:
        key = (row["cell"], row["split"], row["variant_or_contrast"])
        buckets.setdefault(key, []).append(row)
    summaries = []
    for key, block in sorted(buckets.items()):
        for metric in block[0]["values"]:
            values = [row["values"][metric] for row in block]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite reporting metric")
            summaries.append(dict(cell=key[0], split=key[1], variant_or_contrast=key[2],
                metric=metric, n=len(values), mean=mean(values),
                standard_error=stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None))
    return summaries


def collect(task, cells):
    """No generation, fitting or prediction is permitted on this read path."""
    provenance = current_provenance()
    rows = []
    common_inputs = {}
    for cell in cells:
        stores = {}
        for variant in protocol.VARIANTS:
            parameters = dict(cell=cell, variant=variant, shard_index=0,
                              shard_count=1 if task == protocol.PILOT_TASK else 16)
            config = protocol.run_config(task, parameters)
            name = protocol.run_id(task, parameters)
            store = ArtifactStore(OUTPUT_ROOT / name, make_manifest(name, config, provenance))
            store.validate_manifest()
            stores[variant] = store
        for seed in range(1 if task == protocol.PILOT_TASK else 300):
            for split in protocol.SPLITS:
                records = {}
                for variant, store in stores.items():
                    if store.status(seed, split) != "validated":
                        raise ValueError("missing completed scientific cell")
                    record = read_json(store.cell_dir(seed, split) / "result.json")
                    records[variant] = record
                    data_key = (seed, split, variant)
                    inputs = (record["fingerprints"]["training"], record["fingerprints"]["test"],
                              record["evaluation"]["permutation_sha256"])
                    if data_key in common_inputs and common_inputs[data_key] != inputs:
                        raise ValueError("methods did not use identical data and interventions")
                    common_inputs[data_key] = inputs
                validate_pairs(records)
                values = {v: metrics(r) for v, r in records.items()}
                for variant, metric_values in values.items():
                    rows.append(dict(cell=cell, seed=seed, split=split,
                                     variant_or_contrast=variant, values=metric_values))
                for variant in protocol.VARIANTS[1:]:
                    rows.append(dict(cell=cell, seed=seed, split=split,
                        variant_or_contrast=variant + "_minus_control", values={
                            m: values[variant][m] - values["control"][m] for m in values["control"]}))
                rows.append(dict(cell=cell, seed=seed, split=split,
                    variant_or_contrast="factorial_interaction", values={m:
                        values["site_gaussian_local_permutation"][m] - values["site_gaussian_shift"][m]
                        - values["local_permutation"][m] + values["control"][m]
                        for m in values["control"]}))
    return rows


def write_report(task, cells, output_prefix):
    prefix = safe_path(output_prefix)
    csv_path, json_path = Path(str(prefix) + ".csv"), Path(str(prefix) + ".json")
    for path in (csv_path, json_path):
        safe_path(path)
        if path.exists():
            raise FileExistsError("refusing to overwrite comparison report")
    rows = collect(task, cells)
    summaries = aggregate(rows)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    safe_path(prefix)
    atomic_create_text(csv_path, stream.getvalue())
    atomic_create_text(json_path, canonical(dict(task=task, cells=list(cells),
        scope="completed CPU comparison cells only; no historical/GPU timing imputation",
        timing="serial fitting including tuning/refit; evaluation reported separately",
        missing_policy="fail; never silently remove failed/incomplete cells", paired_rows=rows)))
    return csv_path, json_path
