#!/usr/bin/env python3
"""Summarize and visualize SyntheticEffectModifier prediction-intervention results."""

import csv
import html
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "effect_modifier_experiment" / "outputs" / "prediction_intervention_1000seeds"
RESULTS_DIR = ROOT / "results" / "effect_modifier"

METHOD_ORDER = ["fedavg", "fedfisher_diag", "fedfisher_kfac", "pooled"]
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedfisher_diag": "FedFisher diag",
    "fedfisher_kfac": "FedFisher KFAC",
    "pooled": "Pooled",
}
METHOD_COLORS = {
    "fedavg": "#6b7280",
    "fedfisher_diag": "#0072B2",
    "fedfisher_kfac": "#009E73",
    "pooled": "#CC79A7",
}

MODEL_ORDER = ["SyntheticMLP", "SyntheticMLPDeep"]
MODEL_LABELS = {
    "SyntheticMLP": "MLP",
    "SyntheticMLPDeep": "Deep MLP",
}
SPLIT_ORDER = ["iid", "noniid"]
SPLIT_LABELS = {"iid": "IID", "noniid": "non-IID"}
METRIC_ORDER = ["margin_drop", "abs_logit_change"]
METRIC_LABELS = {
    "margin_drop": "Margin drop",
    "abs_logit_change": "Abs logit change",
}


def esc(value):
    return html.escape(str(value), quote=True)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def std_error(values):
    return sample_std(values) / math.sqrt(len(values)) if values else 0.0


def ordered_key(row):
    model = row.get("model", "")
    split = row.get("split", "")
    method = row.get("model_source", "")
    metric = row.get("metric", "")
    return (
        MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
        SPLIT_ORDER.index(split) if split in SPLIT_ORDER else len(SPLIT_ORDER),
        METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER),
        METRIC_ORDER.index(metric) if metric in METRIC_ORDER else len(METRIC_ORDER),
        model,
        split,
        method,
        metric,
    )


def aggregate_csv_files(paths, key_fields, value_fields, row_filter=None):
    buckets = defaultdict(lambda: {field: [] for field in value_fields})
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row_filter and not row_filter(row):
                    continue
                key = tuple(row[field] for field in key_fields)
                for field in value_fields:
                    buckets[key][field].append(float(row[field]))

    rows = []
    for key, values_by_field in buckets.items():
        row = dict(zip(key_fields, key))
        row["n"] = len(next(iter(values_by_field.values())))
        for field, values in values_by_field.items():
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_std"] = sample_std(values)
            row[f"{field}_se"] = std_error(values)
        rows.append(row)
    return sorted(rows, key=ordered_key)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def svg_page(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
        "<style>\n"
        "text{font-family:Arial,Helvetica,sans-serif;letter-spacing:0;fill:#111827}"
        ".title{font-size:22px;font-weight:700}"
        ".subtitle{font-size:12px;fill:#4b5563}"
        ".panel{font-size:14px;font-weight:700}"
        ".axis{font-size:11px;fill:#374151}"
        ".tick{font-size:10px;fill:#4b5563}"
        ".legend{font-size:11px;fill:#111827}"
        ".value{font-size:9px;fill:#111827}"
        ".grid{stroke:#e5e7eb;stroke-width:1}"
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


def linear_scale(domain_min, domain_max, range_min, range_max):
    def scale(value):
        if math.isclose(domain_min, domain_max):
            return (range_min + range_max) / 2
        frac = (value - domain_min) / (domain_max - domain_min)
        return range_min + frac * (range_max - range_min)

    return scale


def draw_legend(body, x, y, methods=METHOD_ORDER):
    cursor = x
    for method in methods:
        body.append(
            f'<rect x="{cursor:.1f}" y="{y - 10:.1f}" width="12" height="12" '
            f'rx="2" fill="{METHOD_COLORS[method]}"/>'
        )
        add_text(body, cursor + 18, y, METHOD_LABELS[method], "legend")
        cursor += 18 + len(METHOD_LABELS[method]) * 6.4 + 20


def draw_y_axis(body, x, y_top, y_bottom, vmin, vmax, ticks, label, tick_fmt):
    y = linear_scale(vmin, vmax, y_bottom, y_top)
    for tick in ticks:
        yy = y(tick)
        body.append(
            f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + 390:.1f}" y2="{yy:.1f}" class="grid"/>'
        )
        add_text(body, x - 7, yy + 3, tick_fmt.format(tick), "tick", "end")
    body.append(
        f'<line x1="{x:.1f}" y1="{y_top:.1f}" x2="{x:.1f}" y2="{y_bottom:.1f}" class="axisline"/>'
    )
    body.append(
        f'<line x1="{x:.1f}" y1="{y_bottom:.1f}" x2="{x + 390:.1f}" y2="{y_bottom:.1f}" class="axisline"/>'
    )
    add_text(
        body,
        x - 43,
        (y_top + y_bottom) / 2,
        label,
        "axis",
        "middle",
        'transform="rotate(-90 {} {})"'.format(x - 43, (y_top + y_bottom) / 2),
    )
    return y


def aggregate_recovery():
    files = sorted(INPUT_DIR.glob("*_prediction_intervention_summary.csv"))
    rows = aggregate_csv_files(
        files,
        ["model_source", "model", "split", "intervention_mode", "metric"],
        [
            "topk_hits",
            "topk_precision",
            "mean_signal_rank",
            "median_signal_rank",
            "mean_noise_rank",
            "auroc",
        ],
        lambda row: row["intervention_mode"] == "permute" and row["metric"] in set(METRIC_ORDER),
    )
    fields = [
        "model_source",
        "model",
        "split",
        "intervention_mode",
        "metric",
        "n",
        "topk_hits_mean",
        "topk_hits_std",
        "topk_hits_se",
        "topk_precision_mean",
        "mean_signal_rank_mean",
        "median_signal_rank_mean",
        "mean_noise_rank_mean",
        "auroc_mean",
        "auroc_std",
        "auroc_se",
    ]
    write_csv(RESULTS_DIR / "effect_modifier_prediction_primary_metrics_summary.csv", rows, fields)
    return rows


def aggregate_model_summary():
    files = sorted(INPUT_DIR.glob("*_prediction_intervention_model_summary.csv"))
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
    for row in rows:
        row["baseline_accuracy_pct_mean"] = row["baseline_accuracy_mean"] * 100.0
        row["baseline_accuracy_pct_std"] = row["baseline_accuracy_std"] * 100.0
        row["baseline_accuracy_pct_se"] = row["baseline_accuracy_se"] * 100.0
    fields = [
        "model_source",
        "model",
        "split",
        "n",
        "baseline_accuracy_pct_mean",
        "baseline_accuracy_pct_std",
        "baseline_accuracy_pct_se",
        "baseline_accuracy_mean",
        "predicted_positive_rate_mean",
        "mean_class1_probability_mean",
        "mean_abs_logit_score_mean",
    ]
    write_csv(RESULTS_DIR / "effect_modifier_prediction_model_summary.csv", rows, fields)
    return rows


def plot_signal_recovery(rows):
    lookup = {
        (row["model"], row["split"], row["model_source"], row["metric"]): row
        for row in rows
    }
    width, height = 1120, 720
    body = []
    add_text(body, 32, 38, "SyntheticEffectModifier signal recovery", "title")
    add_text(
        body,
        32,
        60,
        "1000 seeds; non-IID split; prediction-intervention ranks interaction columns only.",
        "subtitle",
    )
    add_text(
        body,
        32,
        686,
        "Pooled is a centralized reference model, not a federated method.",
        "subtitle",
    )
    draw_legend(body, 585, 54)

    panels = [
        ("A", "SyntheticMLP", "topk_hits", "Top-10 signal hits", 0, 10, [0, 2, 4, 6, 8, 10], "{:.0f}"),
        ("B", "SyntheticMLPDeep", "topk_hits", "Top-10 signal hits", 0, 10, [0, 2, 4, 6, 8, 10], "{:.0f}"),
        ("C", "SyntheticMLP", "auroc", "AUROC", 0.75, 1.0, [0.75, 0.80, 0.85, 0.90, 0.95, 1.00], "{:.2f}"),
        ("D", "SyntheticMLPDeep", "auroc", "AUROC", 0.75, 1.0, [0.75, 0.80, 0.85, 0.90, 0.95, 1.00], "{:.2f}"),
    ]
    panel_positions = [(95, 110), (610, 110), (95, 405), (610, 405)]
    panel_w, panel_h = 390, 220
    bar_w = 24

    for (panel, model, value_name, y_label, vmin, vmax, ticks, tick_fmt), (px, py) in zip(panels, panel_positions):
        add_text(body, px - 48, py - 16, panel, "panel")
        add_text(body, px + panel_w / 2, py - 16, MODEL_LABELS[model], "panel", "middle")
        y = draw_y_axis(body, px, py, py + panel_h, vmin, vmax, ticks, y_label, tick_fmt)
        group_w = panel_w / len(METRIC_ORDER)
        for metric_i, metric in enumerate(METRIC_ORDER):
            center = px + group_w * (metric_i + 0.5)
            for method_i, method in enumerate(METHOD_ORDER):
                row = lookup[(model, "noniid", method, metric)]
                value = float(row[f"{value_name}_mean"])
                se = float(row.get(f"{value_name}_se", 0.0))
                x = center + (method_i - 1.5) * (bar_w + 8)
                yy = y(value)
                body.append(
                    f'<rect x="{x - bar_w / 2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
                    f'height="{py + panel_h - yy:.1f}" fill="{METHOD_COLORS[method]}" rx="2"/>'
                )
                err_top = y(min(vmax, value + se))
                err_bottom = y(max(vmin, value - se))
                body.append(
                    f'<line x1="{x:.1f}" y1="{err_top:.1f}" x2="{x:.1f}" y2="{err_bottom:.1f}" '
                    f'stroke="#111827" stroke-width="1"/>'
                )
                body.append(
                    f'<line x1="{x - 4:.1f}" y1="{err_top:.1f}" x2="{x + 4:.1f}" y2="{err_top:.1f}" '
                    f'stroke="#111827" stroke-width="1"/>'
                )
                value_text = f"{value:.1f}" if value_name == "topk_hits" else f"{value:.2f}"
                add_text(body, x, yy - 6, value_text, "value", "middle")
            add_text(body, center, py + panel_h + 30, METRIC_LABELS[metric], "axis", "middle")

    (RESULTS_DIR / "effect_modifier_signal_recovery.svg").write_text(svg_page(width, height, body))


def plot_model_accuracy(rows):
    lookup = {
        (row["model"], row["split"], row["model_source"]): row
        for row in rows
    }
    width, height = 1120, 600
    body = []
    add_text(body, 32, 38, "SyntheticEffectModifier model accuracy", "title")
    add_text(
        body,
        32,
        60,
        "Held-out test accuracy averaged across 1000 seeds; same trained models used for prediction intervention.",
        "subtitle",
    )
    draw_legend(body, 585, 54)

    panel_positions = [(95, 115), (610, 115)]
    panel_w, panel_h = 390, 335
    bar_w = 24
    for panel_idx, (model, (px, py)) in enumerate(zip(MODEL_ORDER, panel_positions)):
        add_text(body, px - 48, py - 18, chr(ord("A") + panel_idx), "panel")
        add_text(body, px + panel_w / 2, py - 18, MODEL_LABELS[model], "panel", "middle")
        y = draw_y_axis(
            body,
            px,
            py,
            py + panel_h,
            55,
            76,
            [55, 60, 65, 70, 75],
            "Accuracy (%)",
            "{:.0f}",
        )
        group_w = panel_w / len(SPLIT_ORDER)
        for split_i, split in enumerate(SPLIT_ORDER):
            center = px + group_w * (split_i + 0.5)
            for method_i, method in enumerate(METHOD_ORDER):
                row = lookup[(model, split, method)]
                value = float(row["baseline_accuracy_pct_mean"])
                se = float(row["baseline_accuracy_pct_se"])
                x = center + (method_i - 1.5) * (bar_w + 8)
                yy = y(value)
                body.append(
                    f'<rect x="{x - bar_w / 2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" '
                    f'height="{py + panel_h - yy:.1f}" fill="{METHOD_COLORS[method]}" rx="2"/>'
                )
                err_top = y(min(76, value + se))
                err_bottom = y(max(55, value - se))
                body.append(
                    f'<line x1="{x:.1f}" y1="{err_top:.1f}" x2="{x:.1f}" y2="{err_bottom:.1f}" '
                    f'stroke="#111827" stroke-width="1"/>'
                )
                body.append(
                    f'<line x1="{x - 4:.1f}" y1="{err_top:.1f}" x2="{x + 4:.1f}" y2="{err_top:.1f}" '
                    f'stroke="#111827" stroke-width="1"/>'
                )
                add_text(body, x, yy - 6, f"{value:.1f}", "value", "middle")
            add_text(body, center, py + panel_h + 30, SPLIT_LABELS[split], "axis", "middle")

    (RESULTS_DIR / "effect_modifier_model_accuracy.svg").write_text(svg_page(width, height, body))


def write_readme():
    text = """# SyntheticEffectModifier visualizations

Generated by `scripts/make_effect_modifier_visualizations.py` from
`effect_modifier_experiment/outputs/prediction_intervention_1000seeds`.

Files:

- `effect_modifier_signal_recovery.svg`: non-IID prediction-intervention signal recovery for `permute + margin_drop` and `permute + abs_logit_change`.
- `effect_modifier_model_accuracy.svg`: held-out model accuracy for IID and non-IID splits.
- `effect_modifier_prediction_primary_metrics_summary.csv`: aggregated prediction-intervention recovery metrics.
- `effect_modifier_prediction_model_summary.csv`: aggregated model accuracy and score statistics.

Raw per-seed CSVs and logs stay under the ignored `effect_modifier_experiment/outputs/`
and `effect_modifier_experiment/logs/` directories.
"""
    (RESULTS_DIR / "README.md").write_text(text)


def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"missing input directory: {INPUT_DIR}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recovery_rows = aggregate_recovery()
    model_rows = aggregate_model_summary()
    plot_signal_recovery(recovery_rows)
    plot_model_accuracy(model_rows)
    write_readme()
    print(f"Wrote SyntheticEffectModifier visualizations to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
