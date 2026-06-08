#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


FILENAME_RE = re.compile(
    r"one_shot_results_seed(?P<seed>\d+)_SyntheticBinary_(?P<model>[^_]+)"
    r"_epochs(?P<epochs>\d+)_alpha(?P<alpha>[^_]+)_clients(?P<clients>\d+)"
    r"_rounds(?P<rounds>\d+)_split(?P<split>[^_]+)_train(?P<train>\d+)"
    r"_test(?P<test>\d+)_dim(?P<dim>\d+)_sdim(?P<sdim>\d+)"
    r"_sig(?P<signal>[^_]+)_noise(?P<noise>[^_]+)\.csv$"
)

METHOD_ORDER = ["fedavg", "fedfisher_diag", "fedfisher_kfac"]
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedfisher_diag": "FedFisher diag",
    "fedfisher_kfac": "FedFisher KFAC",
}
METHOD_COLORS = {
    "fedavg": "#64748b",
    "fedfisher_diag": "#0f766e",
    "fedfisher_kfac": "#b45309",
}
MODEL_ORDER = ["SyntheticMLP", "SyntheticMLPDeep"]
MODEL_LABELS = {
    "SyntheticMLP": "MLP",
    "SyntheticMLPDeep": "Deep MLP",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SyntheticBinary FedFisher alpha sweep results.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[
            "synthetic_binary_experiment/outputs/original_fedfisher_1000seeds",
            "synthetic_binary_experiment/outputs/original_fedfisher_alpha_sweep_1000seeds",
        ],
        help="One or more result directories. Directories are searched recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default="synthetic_binary_experiment/outputs/original_fedfisher_alpha_sweep_1000seeds/figures",
        help="Directory for alpha sweep figures and summary CSV.",
    )
    return parser.parse_args()


def alpha_to_float(tag):
    return float(tag.replace("p", "."))


def alpha_label(level):
    if level == "iid":
        return "IID"
    return "alpha=%g" % level


def level_sort_key(level):
    if level == "iid":
        return (0, 0.0)
    return (1, -float(level))


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def parse_accuracy(value):
    return float(value.strip().strip("[]"))


def read_rows(input_dirs):
    rows = []
    skipped = []
    seen_paths = set()
    for raw_dir in input_dirs:
        root = Path(raw_dir)
        if not root.exists():
            continue
        for path in sorted(root.rglob("one_shot_results_seed*_SyntheticBinary_*.csv")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            meta = match.groupdict()
            seed = int(meta["seed"])
            split = meta["split"]
            alpha = alpha_to_float(meta["alpha"])
            level = "iid" if split == "iid" else alpha
            file_rows = []
            with path.open(newline="") as handle:
                for key, value in csv.reader(handle):
                    suffix = "_test_acc_%d_0" % seed
                    if key.endswith(suffix):
                        method = key[: -len(suffix)]
                        file_rows.append(
                            {
                                "level": level,
                                "split": split,
                                "alpha": "" if split == "iid" else alpha,
                                "seed": seed,
                                "model": meta["model"],
                                "method": method,
                                "accuracy_pct": parse_accuracy(value),
                            }
                        )
            methods = {row["method"] for row in file_rows}
            if not all(method in methods for method in METHOD_ORDER):
                skipped.append(str(path))
                continue
            rows.extend(row for row in file_rows if row["method"] in METHOD_ORDER)
    return rows, skipped


def build_summary(rows):
    grouped = defaultdict(list)
    by_seed = {}
    for row in rows:
        key = (row["level"], row["model"], row["method"])
        grouped[key].append(row["accuracy_pct"])
        by_seed[(row["level"], row["model"], row["method"], row["seed"])] = row["accuracy_pct"]

    summary = []
    for key in sorted(grouped, key=lambda item: (level_sort_key(item[0]), MODEL_ORDER.index(item[1]), METHOD_ORDER.index(item[2]))):
        level, model, method = key
        values = grouped[key]
        gains = []
        wins = 0
        for row in rows:
            if row["level"] != level or row["model"] != model or row["method"] != method:
                continue
            fedavg = by_seed.get((level, model, "fedavg", row["seed"]))
            if fedavg is None:
                continue
            gain = row["accuracy_pct"] - fedavg
            gains.append(gain)
            if method != "fedavg" and gain > 0:
                wins += 1
        summary.append(
            {
                "level": alpha_label(level),
                "level_sort": level_sort_key(level),
                "alpha": "" if level == "iid" else level,
                "model": model,
                "method": method,
                "n": len(values),
                "accuracy_mean_pct": mean(values),
                "accuracy_std_pct": sample_std(values),
                "gain_over_fedavg_mean_pct": mean(gains) if gains else 0.0,
                "gain_over_fedavg_std_pct": sample_std(gains) if gains else 0.0,
                "seed_wins": wins if method != "fedavg" else 0,
                "win_rate": (wins / len(gains)) if gains and method != "fedavg" else 0.0,
            }
        )
    return summary


def write_summary(rows, output):
    headers = [
        "level",
        "alpha",
        "model",
        "method",
        "n",
        "accuracy_mean_pct",
        "accuracy_std_pct",
        "gain_over_fedavg_mean_pct",
        "gain_over_fedavg_std_pct",
        "seed_wins",
        "win_rate",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in headers}
            for key, value in list(out.items()):
                if isinstance(value, float):
                    out[key] = "%.4f" % value
            writer.writerow(out)


class Svg:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">'
            % (width, height, width, height),
            "<style>"
            "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}"
            ".title{font-size:24px;font-weight:700}"
            ".subtitle{font-size:13px;fill:#52606d}"
            ".axis{font-size:11px;fill:#52606d}"
            ".small{font-size:10px;fill:#52606d}"
            ".label{font-size:12px;fill:#334e68}"
            ".grid{stroke:#e4e7eb;stroke-width:1}"
            ".axisline{stroke:#9aa5b1;stroke-width:1}"
            "</style>",
            '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>' % (width, height),
        ]

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, body, cls="axis", anchor="start", rotate=None, weight=None):
        attrs = ['x="%.2f"' % x, 'y="%.2f"' % y, 'class="%s"' % cls, 'text-anchor="%s"' % anchor]
        if rotate is not None:
            attrs.append('transform="rotate(%s %.2f %.2f)"' % (rotate, x, y))
        if weight is not None:
            attrs.append('font-weight="%s"' % weight)
        self.add("<text %s>%s</text>" % (" ".join(attrs), escape(str(body))))

    def line(self, x1, y1, x2, y2, stroke="#9aa5b1", width=1, cls=None, dash=None):
        attrs = [
            'x1="%.2f"' % x1,
            'y1="%.2f"' % y1,
            'x2="%.2f"' % x2,
            'y2="%.2f"' % y2,
            'stroke="%s"' % stroke,
            'stroke-width="%.2f"' % width,
        ]
        if cls:
            attrs.append('class="%s"' % cls)
        if dash:
            attrs.append('stroke-dasharray="%s"' % dash)
        self.add("<line %s/>" % " ".join(attrs))

    def rect(self, x, y, w, h, fill, stroke=None, radius=0, opacity=None):
        attrs = [
            'x="%.2f"' % x,
            'y="%.2f"' % y,
            'width="%.2f"' % w,
            'height="%.2f"' % h,
            'fill="%s"' % fill,
        ]
        if stroke:
            attrs.append('stroke="%s"' % stroke)
        if radius:
            attrs.append('rx="%.2f" ry="%.2f"' % (radius, radius))
        if opacity is not None:
            attrs.append('opacity="%.2f"' % opacity)
        self.add("<rect %s/>" % " ".join(attrs))

    def circle(self, x, y, r, fill, stroke="#ffffff", width=1):
        self.add(
            '<circle cx="%.2f" cy="%.2f" r="%.2f" fill="%s" stroke="%s" stroke-width="%.2f"/>'
            % (x, y, r, fill, stroke, width)
        )

    def path(self, points, stroke, width=2):
        d = " ".join("%s %.2f %.2f" % ("M" if idx == 0 else "L", x, y) for idx, (x, y) in enumerate(points))
        self.add('<path d="%s" fill="none" stroke="%s" stroke-width="%.2f"/>' % (d, stroke, width))

    def save(self, path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def y_scale(value, min_value, max_value, top, bottom):
    if max_value == min_value:
        return bottom
    return bottom - (value - min_value) * (bottom - top) / (max_value - min_value)


def tick_range(values, step=2, pad=0.5):
    low = min(values) - pad
    high = max(values) + pad
    ymin = math.floor(low / step) * step
    ymax = math.ceil(high / step) * step
    if ymin == ymax:
        ymax += step
    return ymin, ymax


def summary_level_sort_key(label):
    if label == "IID":
        return level_sort_key("iid")
    return level_sort_key(float(label.split("=")[1]))


def n_label(rows):
    ns = sorted({row["n"] for row in rows})
    if not ns:
        return "0"
    if ns[0] == ns[-1]:
        return str(ns[0])
    return "%d-%d" % (ns[0], ns[-1])


def draw_legend(svg, x, y, methods):
    cur = x
    for method in methods:
        svg.rect(cur, y - 10, 14, 14, METHOD_COLORS[method], radius=2)
        svg.text(cur + 20, y + 1, METHOD_LABELS[method], "label")
        cur += 155


def draw_accuracy_bars(summary, output):
    width, height = 1440, 760
    left, right, top, bottom = 85, 45, 125, 620
    panel_gap = 45
    plot_width = width - left - right
    panel_width = (plot_width - panel_gap) / len(MODEL_ORDER)
    svg = Svg(width, height)
    svg.text(40, 42, "Synthetic alpha sweep accuracy", "title")
    svg.text(
        40,
        66,
        "Mean +/- std over %s completed seed/config runs; bars are absolute test accuracy." % n_label(summary),
        "subtitle",
    )
    draw_legend(svg, 825, 52, METHOD_ORDER)

    levels = sorted({row["level"] for row in summary}, key=summary_level_sort_key)
    by_key = {(row["level"], row["model"], row["method"]): row for row in summary}
    bounds = []
    for row in summary:
        bounds.extend([row["accuracy_mean_pct"] - row["accuracy_std_pct"], row["accuracy_mean_pct"] + row["accuracy_std_pct"]])
    ymin, ymax = tick_range(bounds, step=5, pad=2)

    for tick in range(int(ymin), int(ymax) + 1, 5):
        y = y_scale(tick, ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, tick, "axis", "end")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.text(32, 375, "accuracy (%)", "axis", "middle", rotate=-90)

    for model_idx, model in enumerate(MODEL_ORDER):
        panel_left = left + model_idx * (panel_width + panel_gap)
        panel_right = panel_left + panel_width
        panel_center = panel_left + panel_width / 2
        if model_idx:
            svg.line(panel_left - panel_gap / 2, top - 20, panel_left - panel_gap / 2, bottom + 38, "#d9e2ec", 1)
        svg.text(panel_center, top - 24, MODEL_LABELS.get(model, model), "label", "middle", weight="700")
        group_width = panel_width / len(levels)
        bar_slot_width = min(27, (group_width - 22) / len(METHOD_ORDER))
        bar_width = bar_slot_width - 5
        cluster_width = bar_slot_width * len(METHOD_ORDER)
        for level_idx, level in enumerate(levels):
            group_center = panel_left + group_width * level_idx + group_width / 2
            for method_idx, method in enumerate(METHOD_ORDER):
                row = by_key.get((level, model, method))
                if row is None:
                    continue
                value = row["accuracy_mean_pct"]
                std = row["accuracy_std_pct"]
                x = group_center - cluster_width / 2 + method_idx * bar_slot_width + 2.5
                y = y_scale(value, ymin, ymax, top, bottom)
                svg.rect(x, y, bar_width, bottom - y, METHOD_COLORS[method], radius=2)
                err_top = y_scale(value + std, ymin, ymax, top, bottom)
                err_bottom = y_scale(value - std, ymin, ymax, top, bottom)
                err_x = x + bar_width / 2
                svg.line(err_x, err_top, err_x, err_bottom, "#334e68", 1.2)
                svg.line(err_x - 4.5, err_top, err_x + 4.5, err_top, "#334e68", 1.2)
                svg.line(err_x - 4.5, err_bottom, err_x + 4.5, err_bottom, "#334e68", 1.2)
                svg.text(err_x, y - 8, "%.1f" % value, "small", "middle")
            svg.text(group_center, bottom + 25, level, "small", "middle")
        svg.line(panel_right, bottom, panel_right, bottom + 6, "#9aa5b1", 1)

    svg.save(output)


def draw_metric_curve(summary, output, metric, title, subtitle, y_label, y_step, percent=False):
    rows = [row for row in summary if row["method"] != "fedavg"]
    levels = sorted({row["level"] for row in rows}, key=summary_level_sort_key)
    width, height = 1060, 620
    left, right, top, bottom = 85, 40, 95, 500
    svg = Svg(width, height)
    svg.text(40, 42, title, "title")
    svg.text(40, 66, subtitle, "subtitle")

    values = [row[metric] * 100 if percent else row[metric] for row in rows]
    ymin, ymax = tick_range(values, step=y_step, pad=y_step / 3)
    if metric == "gain_over_fedavg_mean_pct":
        ymin = min(ymin, -1)
        ymax = max(ymax, 1)
    for tick in range(int(ymin), int(ymax) + 1, int(y_step)):
        y = y_scale(tick, ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, tick, "axis", "end")
    zero_y = y_scale(0, ymin, ymax, top, bottom)
    svg.line(left, zero_y, width - right, zero_y, stroke="#64748b", width=1.2, dash="4 4")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.text(30, 310, y_label, "axis", "middle", rotate=-90)

    x_positions = {}
    plot_width = width - left - right
    for idx, level in enumerate(levels):
        x = left + (plot_width * idx / max(len(levels) - 1, 1))
        x_positions[level] = x
        svg.text(x, bottom + 24, level, "axis", "middle", rotate=0)

    by_key = {(row["level"], row["model"], row["method"]): row for row in rows}
    for model_idx, model in enumerate(MODEL_ORDER):
        for method in ["fedfisher_diag", "fedfisher_kfac"]:
            points = []
            for level in levels:
                row = by_key.get((level, model, method))
                if row is None:
                    continue
                value = row[metric] * 100 if percent else row[metric]
                points.append((x_positions[level], y_scale(value, ymin, ymax, top, bottom)))
            if not points:
                continue
            stroke = METHOD_COLORS[method]
            line_width = 3 if model == "SyntheticMLPDeep" else 2
            svg.path(points, stroke, width=line_width)
            for x, y in points:
                svg.circle(x, y, 4, stroke)
            legend_x = 650
            legend_y = 48 + model_idx * 44 + (0 if method == "fedfisher_diag" else 18)
            svg.line(legend_x, legend_y - 4, legend_x + 26, legend_y - 4, stroke=stroke, width=line_width)
            svg.text(legend_x + 34, legend_y, "%s %s" % (MODEL_LABELS[model], METHOD_LABELS[method].replace("FedFisher ", "")), "small")

    svg.save(output)


def write_readme(summary, skipped, output):
    levels = sorted({row["level"] for row in summary}, key=lambda label: level_sort_key("iid" if label == "IID" else float(label.split("=")[1])))
    output.write_text(
        "# Synthetic Alpha Sweep Visualizations\n\n"
        "Levels included: %s.\n\n"
        "Files:\n\n"
        "- `alpha_sweep_summary.csv`: numeric table grouped by heterogeneity level, model, and method.\n"
        "- `alpha_accuracy_bars.svg`: absolute test accuracy mean +/- std for all levels and methods.\n"
        "- `alpha_gain_curves.svg`: mean FedFisher gain over FedAvg across IID and alpha levels.\n"
        "- `alpha_win_rate_curves.svg`: fraction of seeds where FedFisher beats FedAvg.\n\n"
        "Skipped incomplete CSV files: %d.\n"
        % (", ".join(levels), len(skipped))
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = read_rows(args.input_dirs)
    if not rows:
        raise RuntimeError("No SyntheticBinary result CSV files found")
    summary = build_summary(rows)
    write_summary(summary, output_dir / "alpha_sweep_summary.csv")
    draw_accuracy_bars(summary, output_dir / "alpha_accuracy_bars.svg")
    draw_metric_curve(
        summary,
        output_dir / "alpha_gain_curves.svg",
        metric="gain_over_fedavg_mean_pct",
        title="FedFisher gain over FedAvg across non-IID levels",
        subtitle="Positive values mean higher test accuracy than one-shot FedAvg; units are accuracy points.",
        y_label="gain over FedAvg (accuracy points)",
        y_step=2,
    )
    draw_metric_curve(
        summary,
        output_dir / "alpha_win_rate_curves.svg",
        metric="win_rate",
        title="FedFisher seed win rate across non-IID levels",
        subtitle="Win rate is the fraction of paired seeds where FedFisher test accuracy exceeds FedAvg.",
        y_label="seed win rate (%)",
        y_step=20,
        percent=True,
    )
    write_readme(summary, skipped, output_dir / "README.md")
    print("Wrote alpha sweep figures to %s" % output_dir)
    if skipped:
        print("Skipped %d incomplete CSV files" % len(skipped))


if __name__ == "__main__":
    main()
