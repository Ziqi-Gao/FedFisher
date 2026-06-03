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

MODEL_ORDER = ["SyntheticMLP", "SyntheticMLPDeep"]
SPLIT_ORDER = ["iid", "noniid"]
METHOD_ORDER = ["fedavg", "fedfisher_diag", "fedfisher_kfac"]
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedfisher_diag": "FedFisher diag",
    "fedfisher_kfac": "FedFisher KFAC",
}
MODEL_LABELS = {
    "SyntheticMLP": "MLP",
    "SyntheticMLPDeep": "Deep MLP",
}
SPLIT_LABELS = {"iid": "IID", "noniid": "non-IID"}
METHOD_COLORS = {
    "fedavg": "#64748b",
    "fedfisher_diag": "#0f766e",
    "fedfisher_kfac": "#b45309",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot original FedFisher synthetic results.")
    parser.add_argument(
        "--input-dir",
        default="synthetic_binary_experiment/outputs/original_fedfisher",
        help="Directory containing one_shot_results_seed*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="synthetic_binary_experiment/outputs/original_fedfisher/figures",
        help="Directory for SVG figures and comparison CSV.",
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


def setting_key(model, split):
    model_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    split_idx = SPLIT_ORDER.index(split) if split in SPLIT_ORDER else len(SPLIT_ORDER)
    return model_idx, split_idx, model, split


def method_key(method):
    idx = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return idx, method


def read_seed_rows(input_dir):
    rows = []
    for path in sorted(Path(input_dir).glob("one_shot_results_seed*_SyntheticBinary_*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        seed = int(meta["seed"])
        with path.open(newline="") as handle:
            for key, value in csv.reader(handle):
                suffix = "_test_acc_%d_0" % seed
                if key.endswith(suffix):
                    rows.append(
                        {
                            "seed": seed,
                            "model": meta["model"],
                            "split": meta["split"],
                            "method": key[: -len(suffix)],
                            "accuracy_pct": parse_accuracy(value),
                        }
                    )
    return rows


def build_compare_rows(seed_rows):
    grouped = defaultdict(list)
    by_seed = {}
    for row in seed_rows:
        grouped[(row["model"], row["split"], row["method"])].append(row["accuracy_pct"])
        by_seed[(row["model"], row["split"], row["method"], row["seed"])] = row["accuracy_pct"]

    compare = []
    for key in sorted(grouped, key=lambda item: (setting_key(item[0], item[1]), method_key(item[2]))):
        model, split, method = key
        values = grouped[key]
        gains = []
        wins = 0
        for row in seed_rows:
            if row["model"] != model or row["split"] != split or row["method"] != method:
                continue
            fedavg = by_seed.get((model, split, "fedavg", row["seed"]))
            if fedavg is None:
                continue
            gain = row["accuracy_pct"] - fedavg
            gains.append(gain)
            if method != "fedavg" and gain > 0:
                wins += 1
        compare.append(
            {
                "model": model,
                "split": split,
                "method": method,
                "n": len(values),
                "accuracy_mean_pct": mean(values),
                "accuracy_std_pct": sample_std(values),
                "gain_over_fedavg_mean_pct": mean(gains) if gains else 0.0,
                "gain_over_fedavg_std_pct": sample_std(gains) if gains else 0.0,
                "seed_wins": wins if method != "fedavg" else 0,
            }
        )
    return compare


def write_compare_csv(rows, output):
    headers = [
        "model",
        "split",
        "method",
        "n",
        "accuracy_mean_pct",
        "accuracy_std_pct",
        "gain_over_fedavg_mean_pct",
        "gain_over_fedavg_std_pct",
        "seed_wins",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in out.items():
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

    def circle(self, x, y, r, fill, stroke=None, width=1):
        attrs = ['cx="%.2f"' % x, 'cy="%.2f"' % y, 'r="%.2f"' % r, 'fill="%s"' % fill]
        if stroke:
            attrs.append('stroke="%s" stroke-width="%.2f"' % (stroke, width))
        self.add("<circle %s/>" % " ".join(attrs))

    def path(self, points, stroke, width=2, fill="none"):
        d = " ".join("%s %.2f %.2f" % ("M" if idx == 0 else "L", x, y) for idx, (x, y) in enumerate(points))
        self.add('<path d="%s" fill="%s" stroke="%s" stroke-width="%.2f"/>' % (d, fill, stroke, width))

    def save(self, path):
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def y_scale(value, min_value, max_value, top, bottom):
    if max_value == min_value:
        return bottom
    return bottom - (value - min_value) * (bottom - top) / (max_value - min_value)


def draw_legend(svg, x, y, methods):
    cur = x
    for method in methods:
        svg.rect(cur, y - 10, 14, 14, METHOD_COLORS[method], radius=2)
        svg.text(cur + 20, y + 1, METHOD_LABELS[method], "label")
        cur += 150


def draw_accuracy_bars(compare_rows, output):
    width, height = 1160, 720
    left, right, top, bottom = 85, 40, 105, 585
    svg = Svg(width, height)
    svg.text(40, 42, "Original FedFisher pipeline synthetic accuracy", "title")
    svg.text(40, 66, "Mean +/- std over 5 seeds; same synthetic data, original FedFisher aggregation path.", "subtitle")
    draw_legend(svg, 620, 52, METHOD_ORDER)

    settings = sorted({(row["model"], row["split"]) for row in compare_rows}, key=lambda item: setting_key(*item))
    by_key = {(row["model"], row["split"], row["method"]): row for row in compare_rows}
    ymin, ymax = 55.0, 76.0
    for tick in range(55, 77, 5):
        y = y_scale(tick, ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, tick, "axis", "end")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.text(32, 360, "accuracy (%)", "axis", "middle", rotate=-90)

    group_w = (width - left - right) / len(settings)
    bar_w = 32
    for group_idx, (model, split) in enumerate(settings):
        cx = left + group_w * group_idx + group_w / 2
        for method_idx, method in enumerate(METHOD_ORDER):
            row = by_key[(model, split, method)]
            value = row["accuracy_mean_pct"]
            std = row["accuracy_std_pct"]
            x = cx - 1.5 * bar_w + method_idx * bar_w + 4
            y = y_scale(value, ymin, ymax, top, bottom)
            svg.rect(x, y, bar_w - 8, bottom - y, METHOD_COLORS[method], radius=2)
            err_top = y_scale(value + std, ymin, ymax, top, bottom)
            err_bottom = y_scale(value - std, ymin, ymax, top, bottom)
            err_x = x + (bar_w - 8) / 2
            svg.line(err_x, err_top, err_x, err_bottom, "#334e68", 1.2)
            svg.line(err_x - 5, err_top, err_x + 5, err_top, "#334e68", 1.2)
            svg.line(err_x - 5, err_bottom, err_x + 5, err_bottom, "#334e68", 1.2)
            svg.text(x + (bar_w - 8) / 2, y - 8, "%.1f" % value, "small", "middle")
        svg.text(cx, bottom + 24, "%s %s" % (MODEL_LABELS.get(model, model), SPLIT_LABELS.get(split, split)), "label", "middle")
    svg.save(output)


def gain_color(value):
    if value >= 0:
        strength = min(abs(value) / 9.0, 1.0)
        r = int(232 - 202 * strength)
        g = int(245 - 80 * strength)
        b = int(233 - 120 * strength)
    else:
        strength = min(abs(value) / 2.0, 1.0)
        r = int(254 - 74 * strength)
        g = int(226 - 98 * strength)
        b = int(226 - 64 * strength)
    return "#%02x%02x%02x" % (r, g, b)


def draw_gain_heatmap(compare_rows, output):
    rows = [row for row in compare_rows if row["method"] != "fedavg"]
    settings = sorted({(row["model"], row["split"]) for row in rows}, key=lambda item: setting_key(*item))
    methods = ["fedfisher_diag", "fedfisher_kfac"]
    width, height = 940, 420
    left, top = 185, 105
    cell_w, cell_h = 175, 82
    svg = Svg(width, height)
    svg.text(40, 42, "FedFisher gain over FedAvg", "title")
    svg.text(40, 66, "Positive values mean higher test accuracy than one-shot FedAvg; units are accuracy points.", "subtitle")
    by_key = {(row["model"], row["split"], row["method"]): row for row in rows}
    for col, setting in enumerate(settings):
        x = left + col * cell_w + cell_w / 2
        svg.text(x, top - 28, MODEL_LABELS.get(setting[0], setting[0]), "label", "middle", weight="700")
        svg.text(x, top - 10, SPLIT_LABELS.get(setting[1], setting[1]), "small", "middle")
    for row_idx, method in enumerate(methods):
        y = top + row_idx * cell_h + cell_h / 2
        svg.text(left - 18, y + 4, METHOD_LABELS[method], "label", "end")
        for col, setting in enumerate(settings):
            row = by_key[(setting[0], setting[1], method)]
            gain = row["gain_over_fedavg_mean_pct"]
            x = left + col * cell_w
            yy = top + row_idx * cell_h
            svg.rect(x, yy, cell_w - 10, cell_h - 10, gain_color(gain), stroke="#ffffff", radius=4)
            svg.text(x + (cell_w - 10) / 2, yy + 35, "%+.2f" % gain, "label", "middle", weight="700")
            svg.text(x + (cell_w - 10) / 2, yy + 54, "wins %d/5" % row["seed_wins"], "small", "middle")
    svg.save(output)


def draw_seed_pairs(seed_rows, output):
    width, height = 1120, 720
    left, right, top, bottom = 95, 50, 100, 600
    svg = Svg(width, height)
    svg.text(40, 42, "Seed-level paired comparison", "title")
    svg.text(40, 66, "Each line connects FedAvg to FedFisher for the same model, split, and seed.", "subtitle")
    settings = sorted({(row["model"], row["split"]) for row in seed_rows}, key=lambda item: setting_key(*item))
    by_seed = {(row["model"], row["split"], row["method"], row["seed"]): row["accuracy_pct"] for row in seed_rows}
    ymin, ymax = 45.0, 78.0
    for tick in range(45, 79, 5):
        y = y_scale(tick, ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, tick, "axis", "end")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.text(34, 350, "accuracy (%)", "axis", "middle", rotate=-90)
    group_w = (width - left - right) / len(settings)
    for idx, (model, split) in enumerate(settings):
        base_x = left + idx * group_w + group_w / 2
        x_positions = {
            "fedavg": base_x - 48,
            "fedfisher_diag": base_x,
            "fedfisher_kfac": base_x + 48,
        }
        for method, x in x_positions.items():
            svg.text(x, top - 12, METHOD_LABELS[method].replace("FedFisher ", "FF "), "small", "middle")
        for seed in range(5):
            vals = {method: by_seed.get((model, split, method, seed)) for method in METHOD_ORDER}
            if any(value is None for value in vals.values()):
                continue
            points = [(x_positions[method], y_scale(vals[method], ymin, ymax, top, bottom)) for method in METHOD_ORDER]
            color = "#0f766e" if vals["fedfisher_diag"] >= vals["fedavg"] else "#b91c1c"
            svg.path(points, color, width=1.5)
            for method, (x, y) in zip(METHOD_ORDER, points):
                svg.circle(x, y, 4, METHOD_COLORS[method], stroke="#ffffff", width=1)
        svg.text(base_x, bottom + 24, "%s %s" % (MODEL_LABELS.get(model, model), SPLIT_LABELS.get(split, split)), "label", "middle")
    svg.save(output)


def write_readme(compare_rows, output):
    best = max(
        (row for row in compare_rows if row["method"] != "fedavg"),
        key=lambda row: row["gain_over_fedavg_mean_pct"],
    )
    worst = min(
        (row for row in compare_rows if row["method"] != "fedavg"),
        key=lambda row: row["gain_over_fedavg_mean_pct"],
    )
    output.write_text(
        "# Original FedFisher Synthetic Visualizations\n\n"
        "Generated from `synthetic_binary_experiment/outputs/original_fedfisher`.\n\n"
        "Files:\n\n"
        "- `original_compare_summary.csv`: numeric comparison table in accuracy percentage points.\n"
        "- `original_accuracy_bars.svg`: absolute test accuracy mean +/- std over 5 seeds.\n"
        "- `original_gain_heatmap.svg`: FedFisher gain over one-shot FedAvg.\n"
        "- `original_seed_pairs.svg`: paired seed-level FedAvg/FedFisher comparison.\n\n"
        "Largest FedFisher gain: %s %s %s, %+0.2f accuracy points.\n"
        "Smallest FedFisher gain: %s %s %s, %+0.2f accuracy points.\n"
        % (
            best["model"],
            best["split"],
            best["method"],
            best["gain_over_fedavg_mean_pct"],
            worst["model"],
            worst["split"],
            worst["method"],
            worst["gain_over_fedavg_mean_pct"],
        )
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = read_seed_rows(args.input_dir)
    if not seed_rows:
        raise RuntimeError("No original synthetic result CSV files found")
    compare_rows = build_compare_rows(seed_rows)
    write_compare_csv(compare_rows, output_dir / "original_compare_summary.csv")
    draw_accuracy_bars(compare_rows, output_dir / "original_accuracy_bars.svg")
    draw_gain_heatmap(compare_rows, output_dir / "original_gain_heatmap.svg")
    draw_seed_pairs(seed_rows, output_dir / "original_seed_pairs.svg")
    write_readme(compare_rows, output_dir / "README.md")
    print("Wrote original FedFisher figures to %s" % output_dir)


if __name__ == "__main__":
    main()
