#!/usr/bin/env python3
"""Generate synthetic FedFisher comparison SVGs without third-party packages."""

import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "results" / "compare"
SYN = ROOT / "synthetic_binary_experiment" / "outputs"

ORIGINAL_SUMMARY = (
    SYN / "original_fedfisher_1000seeds" / "figures" / "original_compare_summary.csv"
)
ALPHA_SUMMARY = (
    SYN
    / "original_fedfisher_alpha_sweep_1000seeds"
    / "figures"
    / "alpha_sweep_summary.csv"
)
PRED_DIR = SYN / "prediction_intervention_1000seeds"
FI_DIR = SYN / "feature_importance_1000seeds"

METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedfisher_diag": "FedFisher diag",
    "fedfisher_kfac": "FedFisher KFAC",
    "pooled": "Pooled",
}

METHOD_COLORS = {
    "fedavg": "#6b7280",
    "fedfisher_diag": "#2563eb",
    "fedfisher_kfac": "#16a34a",
    "pooled": "#9333ea",
}

METRIC_LABELS = {
    "abs_logit_change": "Abs logit change",
    "margin_drop": "Margin drop",
}

FI_LABELS = {
    "weight_norm": "Weight norm",
    "fisher_weighted": "Fisher weighted",
    "global_fisher_weighted": "Global Fisher",
    "ablation_permute_loss": "Permute loss",
}


def esc(value):
    return html.escape(str(value), quote=True)


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def svg_page(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
        "<style>\n"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}"
        ".title{font-size:22px;font-weight:700}"
        ".subtitle{font-size:12px;fill:#4b5563}"
        ".axis{font-size:11px;fill:#374151}"
        ".tick{font-size:10px;fill:#4b5563}"
        ".legend{font-size:12px;fill:#111827}"
        ".value{font-size:10px;fill:#111827}"
        ".panel{font-size:14px;font-weight:700}"
        ".grid{stroke:#e5e7eb;stroke-width:1}"
        ".zero{stroke:#111827;stroke-width:1.3}"
        ".axisline{stroke:#9ca3af;stroke-width:1}"
        "</style>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )


def add_text(body, x, y, text, cls="", anchor="start", extra=""):
    class_attr = f' class="{cls}"' if cls else ""
    body.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{class_attr} {extra}>'
        f"{esc(text)}</text>"
    )


def nice_ticks(vmin, vmax, count=5):
    if math.isclose(vmin, vmax):
        return [vmin]
    raw = (vmax - vmin) / max(1, count - 1)
    mag = 10 ** math.floor(math.log10(abs(raw)))
    step = min([1, 2, 2.5, 5, 10], key=lambda m: abs(raw - m * mag)) * mag
    start = math.floor(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(round(value, 6))
        value += step
    return ticks


def linear_scale(domain_min, domain_max, range_min, range_max):
    def scale(value):
        if math.isclose(domain_min, domain_max):
            return (range_min + range_max) / 2
        frac = (value - domain_min) / (domain_max - domain_min)
        return range_min + frac * (range_max - range_min)

    return scale


def draw_y_axis(body, x, y_top, y_bottom, vmin, vmax, label, tick_fmt="{:.0f}"):
    y = linear_scale(vmin, vmax, y_bottom, y_top)
    for tick in nice_ticks(vmin, vmax, 6):
        yy = y(tick)
        body.append(
            f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + 820:.1f}" y2="{yy:.1f}" class="grid"/>'
        )
        add_text(body, x - 8, yy + 3, tick_fmt.format(tick), "tick", "end")
    add_text(
        body,
        x - 44,
        (y_top + y_bottom) / 2,
        label,
        "axis",
        "middle",
        'transform="rotate(-90 {} {})"'.format(x - 44, (y_top + y_bottom) / 2),
    )


def draw_legend(body, x, y, keys):
    cursor = x
    for key in keys:
        body.append(
            f'<rect x="{cursor:.1f}" y="{y - 10:.1f}" width="12" height="12" '
            f'rx="2" fill="{METHOD_COLORS[key]}"/>'
        )
        add_text(body, cursor + 18, y, METHOD_LABELS[key], "legend")
        cursor += 18 + len(METHOD_LABELS[key]) * 7.0 + 22


def plot_original_gain(rows):
    methods = ["fedfisher_diag", "fedfisher_kfac"]
    groups = [
        ("SyntheticMLP", "iid", "MLP\nIID"),
        ("SyntheticMLP", "noniid", "MLP\nnon-IID"),
        ("SyntheticMLPDeep", "iid", "Deep MLP\nIID"),
        ("SyntheticMLPDeep", "noniid", "Deep MLP\nnon-IID"),
    ]
    lookup = {(r["model"], r["split"], r["method"]): r for r in rows}
    values = [
        float(lookup[(model, split, method)]["gain_over_fedavg_mean_pct"])
        for model, split, _ in groups
        for method in methods
    ]
    vmin = min(-1.0, min(values) - 0.5)
    vmax = max(8.8, max(values) + 0.8)
    width, height = 1040, 600
    left, right, top, bottom = 110, 55, 90, 480
    plot_w = width - left - right
    body = []
    add_text(body, 32, 36, "Synthetic 1000-seed FedFisher gains", "title")
    add_text(
        body,
        32,
        58,
        "Mean accuracy gain over FedAvg; non-IID gains are the main signal.",
        "subtitle",
    )
    draw_legend(body, 650, 52, methods)
    draw_y_axis(body, left, top, bottom, vmin, vmax, "Accuracy gain vs FedAvg (points)", "{:.0f}")
    y = linear_scale(vmin, vmax, bottom, top)
    zero_y = y(0)
    body.append(
        f'<line x1="{left:.1f}" y1="{zero_y:.1f}" x2="{width - right:.1f}" '
        f'y2="{zero_y:.1f}" class="zero"/>'
    )
    group_w = plot_w / len(groups)
    bar_w = 54
    for i, (model, split, label) in enumerate(groups):
        center = left + group_w * (i + 0.5)
        for j, method in enumerate(methods):
            row = lookup[(model, split, method)]
            value = float(row["gain_over_fedavg_mean_pct"])
            x = center + (j - 0.5) * (bar_w + 14)
            yy = y(value)
            rect_y = min(yy, zero_y)
            rect_h = abs(zero_y - yy)
            body.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" '
                f'height="{max(rect_h, 1):.1f}" fill="{METHOD_COLORS[method]}" rx="3"/>'
            )
            text_y = yy - 7 if value >= 0 else yy + 15
            add_text(body, x, text_y, f"{value:+.2f}", "value", "middle")
        top_label, bottom_label = label.split("\n")
        add_text(body, center, bottom + 34, top_label, "axis", "middle")
        add_text(body, center, bottom + 51, bottom_label, "axis", "middle")
    (COMPARE / "synthetic_original_accuracy_gains.svg").write_text(
        svg_page(width, height, body)
    )


def plot_alpha_sweep(rows):
    methods = ["fedfisher_diag", "fedfisher_kfac"]
    models = ["SyntheticMLP", "SyntheticMLPDeep"]
    levels = ["IID", "alpha=2", "alpha=1", "alpha=0.5", "alpha=0.1", "alpha=0.02"]
    xlabels = ["IID", "2", "1", "0.5", "0.1", "0.02"]
    lookup = {(r["level"], r["model"], r["method"]): r for r in rows}
    values = [
        float(lookup[(level, model, method)]["gain_over_fedavg_mean_pct"])
        for level in levels
        for model in models
        for method in methods
        if (level, model, method) in lookup
    ]
    vmin = min(-1.0, min(values) - 0.5)
    vmax = max(8.8, max(values) + 0.8)
    width, height = 1040, 620
    left, panel_w, gap = 95, 410, 70
    top, bottom = 105, 485
    body = []
    add_text(body, 32, 36, "Alpha sweep: heterogeneity drives FedFisher gains", "title")
    add_text(
        body,
        32,
        58,
        "Gain over FedAvg across IID and Dirichlet alpha levels; smaller alpha means stronger heterogeneity.",
        "subtitle",
    )
    add_text(
        body,
        32,
        584,
        "Note: alpha sweep has incomplete coverage in the final array task; non-IID levels use n=963/964.",
        "subtitle",
    )
    draw_legend(body, 650, 52, methods)
    for panel_i, model in enumerate(models):
        px = left + panel_i * (panel_w + gap)
        add_text(body, px + panel_w / 2, 88, "SyntheticMLP" if model == "SyntheticMLP" else "SyntheticMLPDeep", "panel", "middle")
        y = linear_scale(vmin, vmax, bottom, top)
        x = linear_scale(0, len(levels) - 1, px + 35, px + panel_w - 20)
        for tick in nice_ticks(vmin, vmax, 6):
            yy = y(tick)
            body.append(
                f'<line x1="{px:.1f}" y1="{yy:.1f}" x2="{px + panel_w:.1f}" y2="{yy:.1f}" class="grid"/>'
            )
            if panel_i == 0:
                add_text(body, px - 8, yy + 3, f"{tick:.0f}", "tick", "end")
        zero_y = y(0)
        body.append(
            f'<line x1="{px:.1f}" y1="{zero_y:.1f}" x2="{px + panel_w:.1f}" y2="{zero_y:.1f}" class="zero"/>'
        )
        body.append(
            f'<line x1="{px:.1f}" y1="{top:.1f}" x2="{px:.1f}" y2="{bottom:.1f}" class="axisline"/>'
        )
        body.append(
            f'<line x1="{px:.1f}" y1="{bottom:.1f}" x2="{px + panel_w:.1f}" y2="{bottom:.1f}" class="axisline"/>'
        )
        for i, label in enumerate(xlabels):
            add_text(body, x(i), bottom + 26, label, "tick", "middle")
        for method in methods:
            points = []
            for i, level in enumerate(levels):
                row = lookup[(level, model, method)]
                points.append((x(i), y(float(row["gain_over_fedavg_mean_pct"]))))
            d = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
            body.append(
                f'<polyline points="{d}" fill="none" stroke="{METHOD_COLORS[method]}" '
                f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for xx, yy in points:
                body.append(
                    f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.5" fill="{METHOD_COLORS[method]}" stroke="white" stroke-width="1.5"/>'
                )
    add_text(
        body,
        32,
        (top + bottom) / 2,
        "Accuracy gain vs FedAvg (points)",
        "axis",
        "middle",
        'transform="rotate(-90 32 {})"'.format((top + bottom) / 2),
    )
    add_text(body, 520, 545, "Dirichlet alpha", "axis", "middle")
    (COMPARE / "synthetic_alpha_sweep_accuracy_gains.svg").write_text(
        svg_page(width, height, body)
    )


def aggregate_csv_files(files, key_fields, value_fields, row_filter=None):
    buckets = defaultdict(
        lambda: {field: [] for field in value_fields}
    )
    for path in files:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row_filter and not row_filter(row):
                    continue
                key = tuple(row[field] for field in key_fields)
                for field in value_fields:
                    buckets[key][field].append(float(row[field]))
    out = []
    for key, values in sorted(buckets.items()):
        row = dict(zip(key_fields, key))
        n = len(next(iter(values.values()))) if values else 0
        row["n"] = n
        for field, vals in values.items():
            row[f"{field}_mean"] = mean(vals) if vals else ""
        out.append(row)
    return out


def aggregate_prediction_model():
    files = sorted(PRED_DIR.glob("*_prediction_intervention_model_summary.csv"))
    rows = aggregate_csv_files(
        files,
        ["model_source", "model", "split"],
        [
            "baseline_accuracy",
            "predicted_positive_rate",
            "mean_class1_probability",
            "mean_abs_logit_score",
        ],
    )
    write_csv(
        COMPARE / "synthetic_prediction_model_summary.csv",
        rows,
        [
            "model_source",
            "model",
            "split",
            "n",
            "baseline_accuracy_mean",
            "predicted_positive_rate_mean",
            "mean_class1_probability_mean",
            "mean_abs_logit_score_mean",
        ],
    )
    return rows


def aggregate_prediction_primary():
    files = sorted(PRED_DIR.glob("*_prediction_intervention_summary.csv"))
    keep_metrics = {"abs_logit_change", "margin_drop"}
    rows = aggregate_csv_files(
        files,
        ["model_source", "model", "split", "intervention_mode", "metric"],
        ["topk_hits", "topk_precision", "mean_signal_rank", "mean_noise_rank", "auroc"],
        lambda row: row["intervention_mode"] == "permute" and row["metric"] in keep_metrics,
    )
    write_csv(
        COMPARE / "synthetic_prediction_primary_metrics_summary.csv",
        rows,
        [
            "model_source",
            "model",
            "split",
            "intervention_mode",
            "metric",
            "n",
            "topk_hits_mean",
            "topk_precision_mean",
            "mean_signal_rank_mean",
            "mean_noise_rank_mean",
            "auroc_mean",
        ],
    )
    return rows


def aggregate_feature_importance():
    files = sorted(FI_DIR.glob("*_feature_importance_summary.csv"))
    keep_methods = set(FI_LABELS)
    rows = aggregate_csv_files(
        files,
        ["alg", "model", "split", "method"],
        ["topk_hits", "topk_precision", "mean_signal_rank", "mean_noise_rank", "auroc"],
        lambda row: row["method"] in keep_methods,
    )
    write_csv(
        COMPARE / "synthetic_feature_importance_selected_summary.csv",
        rows,
        [
            "alg",
            "model",
            "split",
            "method",
            "n",
            "topk_hits_mean",
            "topk_precision_mean",
            "mean_signal_rank_mean",
            "mean_noise_rank_mean",
            "auroc_mean",
        ],
    )
    return rows


def plot_prediction_accuracy(rows):
    methods = ["fedavg", "fedfisher_diag", "fedfisher_kfac", "pooled"]
    groups = [
        ("SyntheticMLP", "iid", "MLP\nIID"),
        ("SyntheticMLP", "noniid", "MLP\nnon-IID"),
        ("SyntheticMLPDeep", "iid", "Deep MLP\nIID"),
        ("SyntheticMLPDeep", "noniid", "Deep MLP\nnon-IID"),
    ]
    lookup = {
        (str(r["model"]), str(r["split"]), str(r["model_source"])): r for r in rows
    }
    width, height = 1120, 620
    left, right, top, bottom = 95, 45, 92, 490
    body = []
    add_text(body, 32, 36, "Prediction intervention: model accuracy", "title")
    add_text(
        body,
        32,
        58,
        "Averaged across 1000 seeds; FedFisher mainly helps under non-IID splits.",
        "subtitle",
    )
    draw_legend(body, 610, 52, methods)
    draw_y_axis(body, left, top, bottom, 55, 76, "Accuracy (%)", "{:.0f}")
    y = linear_scale(55, 76, bottom, top)
    group_w = (width - left - right) / len(groups)
    bar_w = 36
    for i, (model, split, label) in enumerate(groups):
        center = left + group_w * (i + 0.5)
        for j, method in enumerate(methods):
            row = lookup[(model, split, method)]
            value = float(row["baseline_accuracy_mean"]) * 100.0
            x = center + (j - 1.5) * (bar_w + 8)
            yy = y(value)
            body.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
                f'height="{bottom - yy:.1f}" fill="{METHOD_COLORS[method]}" rx="3"/>'
            )
            add_text(body, x, yy - 6, f"{value:.1f}", "value", "middle")
        top_label, bottom_label = label.split("\n")
        add_text(body, center, bottom + 34, top_label, "axis", "middle")
        add_text(body, center, bottom + 51, bottom_label, "axis", "middle")
    (COMPARE / "synthetic_prediction_accuracy.svg").write_text(svg_page(width, height, body))


def plot_prediction_signal_recovery(rows):
    methods = ["fedavg", "fedfisher_diag", "fedfisher_kfac", "pooled"]
    models = ["SyntheticMLP", "SyntheticMLPDeep"]
    metrics = ["abs_logit_change", "margin_drop"]
    lookup = {
        (
            str(r["model_source"]),
            str(r["model"]),
            str(r["split"]),
            str(r["metric"]),
        ): r
        for r in rows
    }
    width, height = 1120, 640
    left, panel_w, gap = 90, 455, 65
    top, bottom = 105, 500
    body = []
    add_text(body, 32, 36, "Prediction intervention: signal feature recovery", "title")
    add_text(
        body,
        32,
        58,
        "Non-IID only; y-axis is recovered signal dimensions in the top 10 ranked features.",
        "subtitle",
    )
    draw_legend(body, 610, 52, methods)
    for panel_i, metric in enumerate(metrics):
        px = left + panel_i * (panel_w + gap)
        add_text(body, px + panel_w / 2, 88, METRIC_LABELS[metric], "panel", "middle")
        y = linear_scale(0, 10, bottom, top)
        for tick in [0, 2, 4, 6, 8, 10]:
            yy = y(tick)
            body.append(
                f'<line x1="{px:.1f}" y1="{yy:.1f}" x2="{px + panel_w:.1f}" y2="{yy:.1f}" class="grid"/>'
            )
            if panel_i == 0:
                add_text(body, px - 8, yy + 3, f"{tick}", "tick", "end")
        model_w = panel_w / len(models)
        bar_w = 31
        for i, model in enumerate(models):
            center = px + model_w * (i + 0.5)
            for j, method in enumerate(methods):
                row = lookup[(method, model, "noniid", metric)]
                value = float(row["topk_hits_mean"])
                x = center + (j - 1.5) * (bar_w + 8)
                yy = y(value)
                body.append(
                    f'<rect x="{x - bar_w / 2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
                    f'height="{bottom - yy:.1f}" fill="{METHOD_COLORS[method]}" rx="3"/>'
                )
                add_text(body, x, yy - 6, f"{value:.1f}", "value", "middle")
            add_text(
                body,
                center,
                bottom + 32,
                "SyntheticMLP" if model == "SyntheticMLP" else "SyntheticMLPDeep",
                "axis",
                "middle",
            )
    add_text(
        body,
        32,
        (top + bottom) / 2,
        "Top-10 signal hits",
        "axis",
        "middle",
        'transform="rotate(-90 32 {})"'.format((top + bottom) / 2),
    )
    (COMPARE / "synthetic_prediction_signal_recovery.svg").write_text(
        svg_page(width, height, body)
    )


def heat_color(value, vmin=0.0, vmax=10.0):
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    stops = [
        (0.0, (248, 250, 252)),
        (0.35, (191, 219, 254)),
        (0.70, (96, 165, 250)),
        (1.0, (22, 163, 74)),
    ]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if frac <= b:
            local = 0 if math.isclose(a, b) else (frac - a) / (b - a)
            rgb = tuple(round(ca[i] + (cb[i] - ca[i]) * local) for i in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#16a34a"


def plot_feature_importance_heatmap(rows):
    methods = [
        "weight_norm",
        "fisher_weighted",
        "global_fisher_weighted",
        "ablation_permute_loss",
    ]
    row_keys = []
    for model in ["SyntheticMLP", "SyntheticMLPDeep"]:
        for alg in ["fedavg", "fedfisher_diag", "fedfisher_kfac", "pooled"]:
            row_keys.append((alg, model, "noniid"))
    lookup = {
        (str(r["alg"]), str(r["model"]), str(r["split"]), str(r["method"])): r for r in rows
    }
    width, height = 1040, 620
    left, top = 245, 115
    cell_w, cell_h = 150, 42
    body = []
    add_text(body, 32, 36, "Feature importance: selected signal recovery methods", "title")
    add_text(
        body,
        32,
        58,
        "Non-IID only; cells show average recovered signal dimensions among top 10.",
        "subtitle",
    )
    for j, method in enumerate(methods):
        add_text(body, left + j * cell_w + cell_w / 2, top - 18, FI_LABELS[method], "axis", "middle")
    for i, (alg, model, split) in enumerate(row_keys):
        y = top + i * cell_h
        label = ("MLP" if model == "SyntheticMLP" else "Deep MLP") + " / " + METHOD_LABELS[alg].replace("FedFisher ", "")
        add_text(body, left - 12, y + 27, label, "axis", "end")
        for j, method in enumerate(methods):
            x = left + j * cell_w
            key = (alg, model, split, method)
            if key in lookup:
                value = float(lookup[key]["topk_hits_mean"])
                fill = heat_color(value)
                label_text = f"{value:.2f}"
            else:
                fill = "#f3f4f6"
                label_text = "n/a"
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 3:.1f}" height="{cell_h - 3:.1f}" '
                f'fill="{fill}" stroke="white" stroke-width="1"/>'
            )
            add_text(body, x + cell_w / 2, y + 26, label_text, "value", "middle")
    add_text(body, left, top + len(row_keys) * cell_h + 38, "0", "tick")
    body.append(
        f'<linearGradient id="heat" x1="0%" x2="100%" y1="0%" y2="0%">'
        '<stop offset="0%" stop-color="#f8fafc"/>'
        '<stop offset="35%" stop-color="#bfdbfe"/>'
        '<stop offset="70%" stop-color="#60a5fa"/>'
        '<stop offset="100%" stop-color="#16a34a"/>'
        "</linearGradient>"
    )
    body.append(
        f'<rect x="{left + 28:.1f}" y="{top + len(row_keys) * cell_h + 26:.1f}" '
        f'width="300" height="12" fill="url(#heat)" stroke="#d1d5db"/>'
    )
    add_text(body, left + 342, top + len(row_keys) * cell_h + 38, "10 signal hits", "tick")
    (COMPARE / "synthetic_feature_importance_signal_recovery.svg").write_text(
        svg_page(width, height, body)
    )


def write_synthetic_readme():
    path = COMPARE / "README.md"
    old = path.read_text() if path.exists() else ""
    marker = "\n## Synthetic experiment visualizations\n"
    if marker in old:
        old = old.split(marker)[0].rstrip() + "\n"
    addition = """\


## Synthetic experiment visualizations

Generated by `scripts/make_synthetic_compare_visualizations.py`.

Files:

- `synthetic_original_accuracy_gains.svg`: 1000-seed original FedFisher gain over FedAvg.
- `synthetic_alpha_sweep_accuracy_gains.svg`: gain trend as non-IID heterogeneity changes.
- `synthetic_prediction_accuracy.svg`: latest prediction-intervention model accuracy.
- `synthetic_prediction_signal_recovery.svg`: non-IID prediction-intervention signal recovery.
- `synthetic_feature_importance_signal_recovery.svg`: non-IID feature-importance method comparison.
- `synthetic_prediction_model_summary.csv`: aggregated model accuracy and score statistics.
- `synthetic_prediction_primary_metrics_summary.csv`: aggregated prediction-intervention signal recovery.
- `synthetic_feature_importance_selected_summary.csv`: aggregated selected feature-importance methods.

Interpretation: FedFisher helps mostly under non-IID splits. KFAC is usually strongest for both
accuracy and prediction-intervention signal recovery, while IID gains are near zero or negative.
The alpha sweep figure should be read with the existing caveat that the final alpha-sweep array
task timed out, so some non-IID levels have incomplete coverage.
"""
    path.write_text(old.rstrip() + addition)


def main():
    COMPARE.mkdir(parents=True, exist_ok=True)
    original_rows = read_rows(ORIGINAL_SUMMARY)
    alpha_rows = read_rows(ALPHA_SUMMARY)
    plot_original_gain(original_rows)
    plot_alpha_sweep(alpha_rows)
    prediction_model = aggregate_prediction_model()
    prediction_primary = aggregate_prediction_primary()
    feature_importance = aggregate_feature_importance()
    plot_prediction_accuracy(prediction_model)
    plot_prediction_signal_recovery(prediction_primary)
    plot_feature_importance_heatmap(feature_importance)
    write_synthetic_readme()
    print("Wrote synthetic visualizations to", COMPARE)


if __name__ == "__main__":
    main()
