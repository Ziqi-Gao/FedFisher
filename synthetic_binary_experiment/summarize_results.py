#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SUMMARY_FIELDS = [
    "model_type",
    "split",
    "method",
    "n",
    "accuracy_mean",
    "accuracy_std",
    "loss_mean",
    "gap_to_pool_mean",
    "gain_over_fedavg_mean",
    "uplink_scalars",
    "rounds",
    "round_epochs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize synthetic FedFisher experiment results.")
    parser.add_argument("--input", required=True, help="Path to results.csv")
    parser.add_argument("--output", required=True, help="Path to write summary.csv")
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list)


def sample_std(values: Iterable[float]) -> float:
    values_list = list(values)
    if len(values_list) <= 1:
        return 0.0
    avg = mean(values_list)
    var = sum((value - avg) ** 2 for value in values_list) / (len(values_list) - 1)
    return math.sqrt(var)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_type"], row["split"], row["method"])].append(row)
    return grouped


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped = group_rows(rows)
    summary = []
    for key in sorted(grouped):
        model_type, split, method = key
        group = grouped[key]
        accuracies = [float(row["accuracy"]) for row in group]
        losses = [float(row["loss"]) for row in group]
        gaps = [float(row["gap_to_pool"]) for row in group]
        gains = [float(row["gain_over_fedavg"]) for row in group]
        summary.append(
            {
                "model_type": model_type,
                "split": split,
                "method": method,
                "n": len(group),
                "accuracy_mean": f"{mean(accuracies):.8f}",
                "accuracy_std": f"{sample_std(accuracies):.8f}",
                "loss_mean": f"{mean(losses):.8f}",
                "gap_to_pool_mean": f"{mean(gaps):.8f}",
                "gain_over_fedavg_mean": f"{mean(gains):.8f}",
                "uplink_scalars": group[0]["uplink_scalars"],
                "rounds": group[0]["rounds"],
                "round_epochs": group[0]["round_epochs"],
            }
        )
    return summary


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary = summarize(read_rows(Path(args.input)))
    write_summary(Path(args.output), summary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
