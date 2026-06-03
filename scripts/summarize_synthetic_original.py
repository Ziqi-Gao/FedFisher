#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


FILENAME_RE = re.compile(
    r"one_shot_results_seed(?P<seed>\d+)_SyntheticBinary_(?P<model>[^_]+)"
    r"_epochs(?P<epochs>\d+)_alpha(?P<alpha>[^_]+)_clients(?P<clients>\d+)"
    r"_rounds(?P<rounds>\d+)_split(?P<split>[^_]+)_train(?P<train>\d+)"
    r"_test(?P<test>\d+)_dim(?P<dim>\d+)_sdim(?P<sdim>\d+)"
    r"_sig(?P<signal>[^_]+)_noise(?P<noise>[^_]+)\.csv$"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize original FedFisher synthetic runs.")
    parser.add_argument(
        "--input-dir",
        default="synthetic_binary_experiment/outputs/original_fedfisher",
        help="Directory containing one_shot_results_seed*.csv files.",
    )
    parser.add_argument(
        "--output",
        default="synthetic_binary_experiment/outputs/original_fedfisher/summary.csv",
        help="Path to write summary CSV.",
    )
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def parse_accuracy(value):
    return float(value.strip().strip("[]"))


def read_runs(input_dir):
    grouped = defaultdict(list)
    for path in sorted(Path(input_dir).glob("one_shot_results_seed*_SyntheticBinary_*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        seed = int(meta["seed"])
        model = meta["model"]
        split = meta["split"]
        with path.open(newline="") as handle:
            for key, value in csv.reader(handle):
                suffix = "_test_acc_%d_0" % seed
                if key.endswith(suffix):
                    method = key[: -len(suffix)]
                    grouped[(model, split, method)].append(parse_accuracy(value))
    return grouped


def build_summary(grouped):
    rows = []
    fedavg_by_setting = {
        (model, split): values
        for (model, split, method), values in grouped.items()
        if method == "fedavg"
    }
    for key in sorted(grouped):
        model, split, method = key
        values = grouped[key]
        fedavg_values = fedavg_by_setting.get((model, split), [])
        gains = []
        if len(fedavg_values) == len(values):
            gains = [value - fedavg for value, fedavg in zip(values, fedavg_values)]
        rows.append(
            {
                "model": model,
                "split": split,
                "method": method,
                "n": len(values),
                "accuracy_mean_pct": "%.4f" % mean(values),
                "accuracy_std_pct": "%.4f" % sample_std(values),
                "gain_over_fedavg_mean_pct": "%.4f" % mean(gains) if gains else "",
                "gain_over_fedavg_std_pct": "%.4f" % sample_std(gains) if gains else "",
            }
        )
    return rows


def write_summary(rows, output):
    headers = [
        "model",
        "split",
        "method",
        "n",
        "accuracy_mean_pct",
        "accuracy_std_pct",
        "gain_over_fedavg_mean_pct",
        "gain_over_fedavg_std_pct",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = build_summary(read_runs(args.input_dir))
    write_summary(rows, args.output)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
