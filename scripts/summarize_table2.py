#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


DATASET_ORDER = ["FashionMNIST", "SVHN", "CIFAR10", "CINIC10"]
ALPHA_ORDER = ["0.2", "0.1", "0.05"]
ALG_ORDER = [
    "fedavg",
    "otfusion",
    "pfnm",
    "regmean",
    "dense",
    "fisher_merge",
    "fedfisher_diag",
    "fedfisher_kfac",
]

FILENAME_RE = re.compile(
    r"one_shot_results_seed(?P<seed>\d+)_(?P<dataset>[^_]+)_(?P<model>[^_]+)"
    r"_epochs(?P<epochs>\d+)_alpha(?P<alpha>[^_]+)_clients(?P<clients>\d+)"
    r"_rounds(?P<rounds>\d+)\.csv$"
)
VALUE_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def parse_value(raw):
    match = VALUE_RE.search(raw)
    if not match:
        raise ValueError(f"Could not parse numeric value from {raw!r}")
    return float(match.group(0))


def alpha_from_tag(tag):
    return tag.replace("p", ".")


def mean_std(values):
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/table2")
    parser.add_argument("--output", default="results/table2_summary.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    grouped = defaultdict(list)

    for path in sorted(results_dir.glob("one_shot_results_seed*.csv")):
        file_match = FILENAME_RE.match(path.name)
        if not file_match:
            continue
        dataset = file_match.group("dataset")
        alpha = alpha_from_tag(file_match.group("alpha"))
        seed = file_match.group("seed")

        with path.open(newline="") as fh:
            for key, value in csv.reader(fh):
                for alg in ALG_ORDER:
                    if key == f"{alg}_test_acc_{seed}_0":
                        grouped[(dataset, alpha, alg)].append(parse_value(value))
                        break

    rows = []
    for dataset in DATASET_ORDER:
        for alpha in ALPHA_ORDER:
            for alg in ALG_ORDER:
                values = grouped.get((dataset, alpha, alg), [])
                if not values:
                    rows.append([dataset, alpha, alg, "", "", 0, "MISSING"])
                    continue
                mean, std = mean_std(values)
                rows.append([dataset, alpha, alg, f"{mean:.2f}", f"{std:.2f}", len(values), f"{mean:.2f}+/-{std:.2f}"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "alpha", "algorithm", "mean", "std", "n", "formatted"])
        writer.writerows(rows)

    print(f"Wrote {output}")
    missing = [row for row in rows if row[-1] == "MISSING"]
    if missing:
        print(f"Missing {len(missing)} dataset/alpha/algorithm cells")


if __name__ == "__main__":
    main()
