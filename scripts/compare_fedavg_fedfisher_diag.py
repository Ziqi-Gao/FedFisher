#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape


DATASETS = ["FashionMNIST", "SVHN", "CIFAR10", "CINIC10"]
ALPHAS = ["0.2", "0.1", "0.05"]
ALGORITHMS = ["fedavg", "fedfisher_diag"]

FILENAME_RE = re.compile(
    r"one_shot_results_seed(?P<seed>\d+)_(?P<dataset>[^_]+)_(?P<model>[^_]+)"
    r"_epochs(?P<epochs>\d+)_alpha(?P<alpha>[^_]+)_clients(?P<clients>\d+)"
    r"_rounds(?P<rounds>\d+)\.csv$"
)


def alpha_from_tag(tag):
    return tag.replace("p", ".")


def mean_std(values):
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def read_seed_results(results_dir):
    values = {}
    for path in sorted(Path(results_dir).glob("one_shot_results_seed*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        seed = match.group("seed")
        dataset = match.group("dataset")
        alpha = alpha_from_tag(match.group("alpha"))
        with path.open(newline="") as fh:
            for key, value in csv.reader(fh):
                for alg in ALGORITHMS:
                    if key == "%s_test_acc_%s_0" % (alg, seed):
                        values[(dataset, alpha, alg, int(seed))] = float(value)
    return values


def grouped_from_seed_values(seed_values):
    grouped = {}
    for dataset in DATASETS:
        for alpha in ALPHAS:
            for alg in ALGORITHMS:
                vals = []
                for seed in range(5):
                    key = (dataset, alpha, alg, seed)
                    if key in seed_values:
                        vals.append(seed_values[key])
                if vals:
                    grouped[(dataset, alpha, alg)] = vals
    return grouped


def build_rows(grouped):
    rows = []
    for dataset in DATASETS:
        for alpha in ALPHAS:
            fedavg = grouped[(dataset, alpha, "fedavg")]
            diag = grouped[(dataset, alpha, "fedfisher_diag")]
            fedavg_mean, fedavg_std = mean_std(fedavg)
            diag_mean, diag_std = mean_std(diag)
            delta = diag_mean - fedavg_mean
            rel = 100.0 * delta / fedavg_mean if fedavg_mean else 0.0
            wins = sum(1 for f, d in zip(fedavg, diag) if d > f)
            rows.append(
                {
                    "dataset": dataset,
                    "alpha": alpha,
                    "fedavg_mean": fedavg_mean,
                    "fedavg_std": fedavg_std,
                    "fedfisher_diag_mean": diag_mean,
                    "fedfisher_diag_std": diag_std,
                    "delta": delta,
                    "relative_gain_pct": rel,
                    "seed_wins": wins,
                    "n": len(fedavg),
                }
            )
    return rows


def write_compare_csv(rows, output):
    headers = [
        "dataset",
        "alpha",
        "fedavg_mean",
        "fedavg_std",
        "fedfisher_diag_mean",
        "fedfisher_diag_std",
        "delta",
        "relative_gain_pct",
        "seed_wins",
        "n",
    ]
    with Path(output).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in headers:
                if isinstance(out[key], float):
                    out[key] = "%.4f" % out[key]
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
        attrs = [
            'cx="%.2f"' % x,
            'cy="%.2f"' % y,
            'r="%.2f"' % r,
            'fill="%s"' % fill,
        ]
        if stroke:
            attrs.append('stroke="%s" stroke-width="%.2f"' % (stroke, width))
        self.add("<circle %s/>" % " ".join(attrs))

    def path(self, points, stroke, width=2, fill="none"):
        d = " ".join(
            ("%s %.2f %.2f" % ("M" if i == 0 else "L", x, y))
            for i, (x, y) in enumerate(points)
        )
        self.add('<path d="%s" fill="%s" stroke="%s" stroke-width="%.2f"/>' % (d, fill, stroke, width))

    def save(self, path):
        self.parts.append("</svg>")
        Path(path).write_text("\n".join(self.parts))


def draw_legend(svg, x, y, entries):
    cur = x
    for label, color in entries:
        svg.rect(cur, y - 10, 14, 14, color, radius=2)
        svg.text(cur + 20, y + 1, label, "label")
        cur += 150


def y_scale(value, min_value, max_value, top, bottom):
    if max_value == min_value:
        return bottom
    return bottom - (value - min_value) * (bottom - top) / (max_value - min_value)


def draw_grouped_bars(rows, output):
    width, height = 1320, 760
    left, right, top, bottom = 85, 40, 105, 610
    svg = Svg(width, height)
    svg.text(40, 42, "FedAvg vs FedFisher diag", "title")
    svg.text(40, 66, "Test accuracy mean +/- std over 5 seeds; grouped by dataset and Dirichlet alpha.", "subtitle")
    draw_legend(svg, 930, 52, [("FedAvg", "#64748b"), ("FedFisher diag", "#0f766e")])

    max_value = max(max(r["fedavg_mean"] + r["fedavg_std"], r["fedfisher_diag_mean"] + r["fedfisher_diag_std"]) for r in rows)
    max_value = math.ceil((max_value + 4) / 10.0) * 10
    for tick in range(0, int(max_value) + 1, 10):
        y = y_scale(tick, 0, max_value, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, str(tick), "axis", "end")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.text(22, 355, "Test accuracy (%)", "label", rotate=-90, anchor="middle")

    plot_w = width - left - right
    group_w = plot_w / len(rows)
    bar_w = min(22, group_w * 0.27)
    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        vals = [
            ("FedAvg", row["fedavg_mean"], row["fedavg_std"], "#64748b", -bar_w * 0.6),
            ("FedFisher diag", row["fedfisher_diag_mean"], row["fedfisher_diag_std"], "#0f766e", bar_w * 0.6),
        ]
        for _, mean, std, color, off in vals:
            x = cx + off - bar_w / 2
            y = y_scale(mean, 0, max_value, top, bottom)
            svg.rect(x, y, bar_w, bottom - y, color, radius=3)
            err_top = y_scale(mean + std, 0, max_value, top, bottom)
            err_bot = y_scale(max(mean - std, 0), 0, max_value, top, bottom)
            svg.line(cx + off, err_top, cx + off, err_bot, "#273444", 1.2)
            svg.line(cx + off - 5, err_top, cx + off + 5, err_top, "#273444", 1.2)
            svg.line(cx + off - 5, err_bot, cx + off + 5, err_bot, "#273444", 1.2)
        label = "%s\\nalpha=%s" % (row["dataset"], row["alpha"])
        svg.text(cx, bottom + 24, row["dataset"], "axis", "middle", rotate=-35)
        svg.text(cx, bottom + 52, "a=%s" % row["alpha"], "small", "middle", rotate=-35)
        svg.text(cx, y_scale(max(row["fedavg_mean"], row["fedfisher_diag_mean"]) + 4, 0, max_value, top, bottom),
                 "+%.1f" % row["delta"], "small", "middle")
    svg.text(width / 2, 725, "Each +value is FedFisher diag minus FedAvg accuracy points.", "subtitle", "middle")
    svg.save(output)


def color_for_delta(delta, min_delta, max_delta):
    if max_delta == min_delta:
        t = 1.0
    else:
        t = (delta - min_delta) / (max_delta - min_delta)
    # Light amber to saturated teal.
    r1, g1, b1 = 255, 247, 237
    r2, g2, b2 = 15, 118, 110
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return "#%02x%02x%02x" % (r, g, b)


def draw_heatmap(rows, output):
    width, height = 760, 520
    svg = Svg(width, height)
    svg.text(40, 42, "FedFisher diag improvement over FedAvg", "title")
    svg.text(40, 66, "Absolute test accuracy gain in percentage points.", "subtitle")
    left, top = 150, 115
    cell_w, cell_h = 145, 70
    deltas = [r["delta"] for r in rows]
    min_delta, max_delta = min(deltas), max(deltas)
    by_key = {(r["dataset"], r["alpha"]): r for r in rows}
    for j, alpha in enumerate(ALPHAS):
        svg.text(left + j * cell_w + cell_w / 2, top - 18, "alpha=%s" % alpha, "label", "middle")
    for i, dataset in enumerate(DATASETS):
        svg.text(left - 18, top + i * cell_h + cell_h / 2 + 4, dataset, "label", "end")
        for j, alpha in enumerate(ALPHAS):
            row = by_key[(dataset, alpha)]
            x = left + j * cell_w
            y = top + i * cell_h
            fill = color_for_delta(row["delta"], min_delta, max_delta)
            svg.rect(x, y, cell_w - 6, cell_h - 6, fill, stroke="#ffffff", radius=4)
            svg.text(x + cell_w / 2 - 3, y + 30, "+%.2f" % row["delta"], "label", "middle", weight="700")
            svg.text(x + cell_w / 2 - 3, y + 50, "%d/5 seed wins" % row["seed_wins"], "small", "middle")
    svg.text(40, height - 38, "Darker teal means larger gain. All cells are positive in this run.", "subtitle")
    svg.save(output)


def draw_alpha_lines(rows, output):
    width, height = 1120, 700
    svg = Svg(width, height)
    svg.text(40, 42, "Accuracy vs non-IID severity", "title")
    svg.text(40, 66, "Lower alpha means more heterogeneous client data.", "subtitle")
    draw_legend(svg, 790, 52, [("FedAvg", "#64748b"), ("FedFisher diag", "#0f766e")])
    by_key = {(r["dataset"], r["alpha"]): r for r in rows}
    panel_w, panel_h = 245, 235
    gap_x, gap_y = 25, 50
    start_x, start_y = 80, 115
    all_values = []
    for r in rows:
        all_values.extend([r["fedavg_mean"], r["fedfisher_diag_mean"]])
    ymin = max(0, math.floor((min(all_values) - 5) / 10.0) * 10)
    ymax = math.ceil((max(all_values) + 5) / 10.0) * 10
    x_order = ["0.05", "0.1", "0.2"]
    for idx, dataset in enumerate(DATASETS):
        col = idx % 2
        row_idx = idx // 2
        x0 = start_x + col * (panel_w + gap_x)
        y0 = start_y + row_idx * (panel_h + gap_y)
        svg.text(x0, y0 - 14, dataset, "label", weight="700")
        for tick in range(int(ymin), int(ymax) + 1, 10):
            y = y_scale(tick, ymin, ymax, y0, y0 + panel_h - 40)
            svg.line(x0, y, x0 + panel_w, y, cls="grid")
            svg.text(x0 - 8, y + 4, tick, "small", "end")
        svg.line(x0, y0, x0, y0 + panel_h - 40, cls="axisline")
        svg.line(x0, y0 + panel_h - 40, x0 + panel_w, y0 + panel_h - 40, cls="axisline")
        xs = []
        for i, alpha in enumerate(x_order):
            x = x0 + 28 + i * ((panel_w - 56) / 2.0)
            xs.append(x)
            svg.text(x, y0 + panel_h - 20, alpha, "small", "middle")
        for alg, color, key in [("FedAvg", "#64748b", "fedavg_mean"), ("FedFisher diag", "#0f766e", "fedfisher_diag_mean")]:
            pts = []
            for i, alpha in enumerate(x_order):
                val = by_key[(dataset, alpha)][key]
                pts.append((xs[i], y_scale(val, ymin, ymax, y0, y0 + panel_h - 40)))
            svg.path(pts, color, 2.5)
            for x, y in pts:
                svg.circle(x, y, 4, color, "#ffffff", 1.5)
    svg.text(620, 325, "alpha", "label", "middle")
    svg.text(28, 350, "Test accuracy (%)", "label", rotate=-90, anchor="middle")
    svg.save(output)


def draw_seed_pairs(rows, seed_values, output):
    width, height = 1280, 780
    left, right, top, bottom = 90, 40, 110, 620
    svg = Svg(width, height)
    svg.text(40, 42, "Seed-level paired comparison", "title")
    svg.text(40, 66, "Each line connects FedAvg and FedFisher diag for the same seed.", "subtitle")
    draw_legend(svg, 820, 52, [("FedAvg", "#64748b"), ("FedFisher diag", "#0f766e")])
    max_value = max(max(v for k, v in seed_values.items() if k[2] in ALGORITHMS), 1)
    max_value = math.ceil((max_value + 5) / 10.0) * 10
    for tick in range(0, int(max_value) + 1, 10):
        y = y_scale(tick, 0, max_value, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, str(tick), "axis", "end")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.text(22, 355, "Test accuracy (%)", "label", rotate=-90, anchor="middle")

    group_w = (width - left - right) / len(rows)
    offsets = [-10, -5, 0, 5, 10]
    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        x_fed = cx - 15
        x_diag = cx + 15
        for seed in range(5):
            f = seed_values[(row["dataset"], row["alpha"], "fedavg", seed)]
            d = seed_values[(row["dataset"], row["alpha"], "fedfisher_diag", seed)]
            y_f = y_scale(f, 0, max_value, top, bottom)
            y_d = y_scale(d, 0, max_value, top, bottom)
            xfo = x_fed + offsets[seed] * 0.35
            xdo = x_diag + offsets[seed] * 0.35
            svg.line(xfo, y_f, xdo, y_d, "#cbd2d9", 1.1)
            svg.circle(xfo, y_f, 3.4, "#64748b", "#ffffff", 1)
            svg.circle(xdo, y_d, 3.4, "#0f766e", "#ffffff", 1)
        svg.text(cx, bottom + 25, row["dataset"], "axis", "middle", rotate=-35)
        svg.text(cx, bottom + 53, "a=%s" % row["alpha"], "small", "middle", rotate=-35)
    svg.text(width / 2, 740, "Upward sloping lines indicate FedFisher diag is better for that seed.", "subtitle", "middle")
    svg.save(output)


def write_readme(rows, output):
    best = max(rows, key=lambda r: r["delta"])
    worst = min(rows, key=lambda r: r["delta"])
    lines = [
        "# FedAvg vs FedFisher diag comparison",
        "",
        "Generated from `results/table2/*.csv`.",
        "",
        "Files:",
        "",
        "- `fedavg_vs_fedfisher_diag_summary.csv`: numeric comparison table.",
        "- `fedavg_vs_fedfisher_diag_bars.svg`: grouped mean accuracy bars with std error bars.",
        "- `fedavg_vs_fedfisher_diag_improvement_heatmap.svg`: absolute accuracy gain heatmap.",
        "- `fedavg_vs_fedfisher_diag_alpha_lines.svg`: accuracy trend as alpha changes.",
        "- `fedavg_vs_fedfisher_diag_seed_pairs.svg`: paired seed-level comparison.",
        "",
        "Largest gain: %s alpha=%s, +%.2f accuracy points."
        % (best["dataset"], best["alpha"], best["delta"]),
        "Smallest gain: %s alpha=%s, +%.2f accuracy points."
        % (worst["dataset"], worst["alpha"], worst["delta"]),
    ]
    Path(output).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/table2")
    parser.add_argument("--output-dir", default="results/compare")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_values = read_seed_results(args.results_dir)
    grouped = grouped_from_seed_values(seed_values)
    rows = build_rows(grouped)

    write_compare_csv(rows, out_dir / "fedavg_vs_fedfisher_diag_summary.csv")
    draw_grouped_bars(rows, out_dir / "fedavg_vs_fedfisher_diag_bars.svg")
    draw_heatmap(rows, out_dir / "fedavg_vs_fedfisher_diag_improvement_heatmap.svg")
    draw_alpha_lines(rows, out_dir / "fedavg_vs_fedfisher_diag_alpha_lines.svg")
    draw_seed_pairs(rows, seed_values, out_dir / "fedavg_vs_fedfisher_diag_seed_pairs.svg")
    write_readme(rows, out_dir / "README.md")

    print("Wrote comparison outputs to %s" % out_dir)


if __name__ == "__main__":
    main()
