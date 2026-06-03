#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="fedfisher_synth_smoke_") as tmp:
        output_dir = Path(tmp)
        run(
            [
                sys.executable,
                str(repo_root / "synthetic_binary_experiment" / "run_experiment.py"),
                "--output-dir",
                str(output_dir),
                "--num-train",
                "300",
                "--num-test",
                "300",
                "--local-epochs",
                "2",
                "--batch-size",
                "64",
                "--fisher-batch-size",
                "64",
                "--fisher-server-steps",
                "40",
                "--fisher-server-eval-every",
                "10",
                "--fisher-val-size",
                "64",
                "--seeds",
                "0",
                "--model-types",
                "lr",
                "mlp",
                "--splits",
                "iid",
            ]
        )
        results = output_dir / "results.csv"
        if not results.exists():
            raise AssertionError("results.csv was not created")
        with results.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected_rows = 2 * 4
        if len(rows) != expected_rows:
            raise AssertionError(f"Expected {expected_rows} rows, found {len(rows)}")
        methods = {row["method"] for row in rows}
        if methods != {"pool", "fedavg_oneshot", "fedfisher_diag", "fedfisher_full", "fedfisher_kfac"}:
            raise AssertionError(f"Unexpected methods: {methods}")
        lr_methods = {row["method"] for row in rows if row["model_type"] == "lr"}
        if "fedfisher_full" not in lr_methods:
            raise AssertionError("LR smoke run did not include fedfisher_full")
        mlp_methods = {row["method"] for row in rows if row["model_type"] == "mlp"}
        if "fedfisher_kfac" not in mlp_methods:
            raise AssertionError("MLP smoke run did not include fedfisher_kfac")
        for row in rows:
            acc = float(row["accuracy"])
            if not 0.0 <= acc <= 1.0:
                raise AssertionError(f"Invalid accuracy {acc}")
        for model_type in {"lr", "mlp"}:
            model_rows = [row for row in rows if row["model_type"] == model_type]
            pool_acc = max(float(row["accuracy"]) for row in model_rows if row["method"] == "pool")
            best_federated = max(float(row["accuracy"]) for row in model_rows if row["method"] != "pool")
            if pool_acc < best_federated:
                raise AssertionError(
                    f"{model_type} pooled baseline {pool_acc:.4f} is below federated result {best_federated:.4f}"
                )

        summary = output_dir / "summary.csv"
        run(
            [
                sys.executable,
                str(repo_root / "synthetic_binary_experiment" / "summarize_results.py"),
                "--input",
                str(results),
                "--output",
                str(summary),
            ]
        )
        if not summary.exists():
            raise AssertionError("summary.csv was not created")
    print("Smoke test passed")


if __name__ == "__main__":
    main()
