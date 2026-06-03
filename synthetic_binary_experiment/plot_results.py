#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.sax.saxutils import escape


METHOD_ORDER = ["pool", "fedavg_oneshot", "fedfisher_diag", "fedfisher_full", "fedfisher_kfac"]
FEDFISHER_METHODS = ["fedfisher_diag", "fedfisher_full", "fedfisher_kfac"]
METHOD_LABELS = {
    "pool": "Pool",
    "fedavg_oneshot": "FedAvg",
    "fedfisher_diag": "FedFisher diag",
    "fedfisher_full": "FedFisher full",
    "fedfisher_kfac": "FedFisher KFAC",
}
METHOD_COLORS = {
    "pool": "#94a3b8",
    "fedavg_oneshot": "#64748b",
    "fedfisher_diag": "#0f766e",
    "fedfisher_full": "#2563eb",
    "fedfisher_kfac": "#b45309",
}
MODEL_ORDER = ["lr", "mlp"]
SPLIT_ORDER = ["iid", "noniid", "mild"]
MODEL_LABELS = {"lr": "LR", "mlp": "MLP"}
SPLIT_LABELS = {"iid": "IID", "noniid": "non-IID", "mild": "mild"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic FedFisher experiment results.")
    parser.add_argument(
        "--input",
        default="synthetic_binary_experiment/outputs/main_cpu/results.csv",
        help="Path to results.csv produced by run_experiment.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="synthetic_binary_experiment/outputs/main_cpu/figures",
        help="Directory for SVG figures and compare summary.",
    )
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list)


def sample_std(values: Iterable[float]) -> float:
    values_list = list(values)
    if len(values_list) <= 1:
        return 0.0
    avg = mean(values_list)
    return math.sqrt(sum((value - avg) ** 2 for value in values_list) / (len(values_list) - 1))


def read_results(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def setting_sort_key(setting: Tuple[str, str]) -> Tuple[int, int, str, str]:
    model, split = setting
    model_idx = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    split_idx = SPLIT_ORDER.index(split) if split in SPLIT_ORDER else len(SPLIT_ORDER)
    return model_idx, split_idx, model, split


def method_sort_key(method: str) -> Tuple[int, str]:
    idx = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return idx, method


def group_values(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_type"], row["split"], row["method"])].append(row)
    return grouped


def build_compare_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped = group_values(rows)
    by_seed = {
        (row["model_type"], row["split"], row["method"], int(row["seed"])): float(row["accuracy"])
        for row in rows
    }
    compare_rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: (setting_sort_key(item[:2]), method_sort_key(item[2]))):
        model, split, method = key
        values = [float(row["accuracy"]) * 100.0 for row in grouped[key]]
        seeds = [int(row["seed"]) for row in grouped[key]]
        gap_values = [float(row["gap_to_pool"]) * 100.0 for row in grouped[key]]
        selected_steps = [
            step
            for step in (parse_optional_float(row.get("fisher_selected_step")) for row in grouped[key])
            if step is not None
        ]
        validation_scores = [
            score * 100.0
            for score in (parse_optional_float(row.get("fisher_val_score")) for row in grouped[key])
            if score is not None
        ]
        gains = []
        wins = 0
        for seed, value_pct in zip(seeds, values):
            fedavg_key = (model, split, "fedavg_oneshot", seed)
            if fedavg_key in by_seed:
                fedavg_pct = by_seed[fedavg_key] * 100.0
                gain = value_pct - fedavg_pct
                gains.append(gain)
                if method != "fedavg_oneshot" and value_pct > fedavg_pct:
                    wins += 1
        compare_rows.append(
            {
                "model_type": model,
                "split": split,
                "method": method,
                "accuracy_mean_pct": mean(values),
                "accuracy_std_pct": sample_std(values),
                "gain_over_fedavg_mean_pct": mean(gains) if gains else 0.0,
                "gain_over_fedavg_std_pct": sample_std(gains) if gains else 0.0,
                "gap_to_pool_mean_pct": mean(gap_values),
                "gap_to_pool_std_pct": sample_std(gap_values),
                "seed_wins": wins if method in FEDFISHER_METHODS else 0,
                "selected_step_mean": mean(selected_steps) if validation_scores else "",
                "selected_step_nonzero": sum(1 for step in selected_steps if step > 0) if validation_scores else "",
                "fisher_val_score_mean_pct": mean(validation_scores) if validation_scores else "",
                "n": len(values),
                "uplink_scalars": grouped[key][0]["uplink_scalars"],
            }
        )
    return compare_rows


def write_compare_csv(rows: List[Dict[str, object]], output: Path) -> None:
    headers = [
        "model_type",
        "split",
        "method",
        "accuracy_mean_pct",
        "accuracy_std_pct",
        "gain_over_fedavg_mean_pct",
        "gain_over_fedavg_std_pct",
        "gap_to_pool_mean_pct",
        "gap_to_pool_std_pct",
        "seed_wins",
        "selected_step_mean",
        "selected_step_nonzero",
        "fisher_val_score_mean_pct",
        "n",
        "uplink_scalars",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in headers:
                if isinstance(out[key], float):
                    out[key] = f"{out[key]:.4f}"
            writer.writerow(out)


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
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
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        ]

    def add(self, text: str) -> None:
        self.parts.append(text)

    def text(
        self,
        x: float,
        y: float,
        body: object,
        cls: str = "axis",
        anchor: str = "start",
        rotate: float | None = None,
        weight: str | None = None,
    ) -> None:
        attrs = [f'x="{x:.2f}"', f'y="{y:.2f}"', f'class="{cls}"', f'text-anchor="{anchor}"']
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate} {x:.2f} {y:.2f})"')
        if weight is not None:
            attrs.append(f'font-weight="{weight}"')
        self.add(f"<text {' '.join(attrs)}>{escape(str(body))}</text>")

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#9aa5b1",
        width: float = 1,
        cls: str | None = None,
        dash: str | None = None,
    ) -> None:
        attrs = [
            f'x1="{x1:.2f}"',
            f'y1="{y1:.2f}"',
            f'x2="{x2:.2f}"',
            f'y2="{y2:.2f}"',
            f'stroke="{stroke}"',
            f'stroke-width="{width:.2f}"',
        ]
        if cls:
            attrs.append(f'class="{cls}"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.add(f"<line {' '.join(attrs)}/>")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str | None = None,
        radius: float = 0,
        opacity: float | None = None,
    ) -> None:
        attrs = [
            f'x="{x:.2f}"',
            f'y="{y:.2f}"',
            f'width="{w:.2f}"',
            f'height="{h:.2f}"',
            f'fill="{fill}"',
        ]
        if stroke:
            attrs.append(f'stroke="{stroke}"')
        if radius:
            attrs.append(f'rx="{radius:.2f}" ry="{radius:.2f}"')
        if opacity is not None:
            attrs.append(f'opacity="{opacity:.2f}"')
        self.add(f"<rect {' '.join(attrs)}/>")

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str | None = None, width: float = 1) -> None:
        attrs = [f'cx="{x:.2f}"', f'cy="{y:.2f}"', f'r="{r:.2f}"', f'fill="{fill}"']
        if stroke:
            attrs.append(f'stroke="{stroke}" stroke-width="{width:.2f}"')
        self.add(f"<circle {' '.join(attrs)}/>")

    def path(self, points: List[Tuple[float, float]], stroke: str, width: float = 2, fill: str = "none") -> None:
        d = " ".join(f"{'M' if idx == 0 else 'L'} {x:.2f} {y:.2f}" for idx, (x, y) in enumerate(points))
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}"/>')

    def save(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def draw_legend(svg: Svg, x: float, y: float, methods: Iterable[str], step: float = 145) -> None:
    cur = x
    for method in methods:
        svg.rect(cur, y - 10, 14, 14, METHOD_COLORS[method], radius=2)
        svg.text(cur + 20, y + 1, METHOD_LABELS[method], "label")
        cur += step


def y_scale(value: float, min_value: float, max_value: float, top: float, bottom: float) -> float:
    if max_value == min_value:
        return bottom
    return bottom - (value - min_value) * (bottom - top) / (max_value - min_value)


def accuracy_axis(values: Iterable[float]) -> Tuple[float, float]:
    values_list = list(values)
    ymin = math.floor((min(values_list) - 2.0) / 5.0) * 5.0
    ymax = math.ceil((max(values_list) + 2.0) / 5.0) * 5.0
    return max(0.0, ymin), min(100.0, ymax)


def setting_label(model: str, split: str) -> str:
    return f"{MODEL_LABELS.get(model, model)} {SPLIT_LABELS.get(split, split)}"


def draw_accuracy_bars(compare_rows: List[Dict[str, object]], output: Path) -> None:
    width, height = 1180, 720
    left, right, top, bottom = 85, 40, 105, 585
    svg = Svg(width, height)
    svg.text(40, 42, "Synthetic FedFisher accuracy", "title")
    svg.text(40, 66, "Absolute test accuracy mean +/- std over 5 seeds; FedFisher gains are concentrated in MLP non-IID.", "subtitle")
    draw_legend(svg, 525, 52, METHOD_ORDER, step=125)

    settings = sorted({(str(row["model_type"]), str(row["split"])) for row in compare_rows}, key=setting_sort_key)
    by_key = {(row["model_type"], row["split"], row["method"]): row for row in compare_rows}
    ymin, ymax = 0.0, 100.0
    for tick in range(0, 101, 20):
        y = y_scale(tick, ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, tick, "axis", "end")
    svg.line(left, top, left, bottom, cls="axisline")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.text(22, 350, "Test accuracy (%)", "label", rotate=-90, anchor="middle")

    group_w = (width - left - right) / len(settings)
    for idx, (model, split) in enumerate(settings):
        present = [method for method in METHOD_ORDER if (model, split, method) in by_key]
        cx = left + group_w * (idx + 0.5)
        bar_w = min(28.0, group_w / (len(present) + 2))
        start_x = cx - bar_w * len(present) / 2.0
        for method_idx, method in enumerate(present):
            row = by_key[(model, split, method)]
            mean_acc = float(row["accuracy_mean_pct"])
            std_acc = float(row["accuracy_std_pct"])
            x = start_x + method_idx * bar_w
            y = y_scale(mean_acc, ymin, ymax, top, bottom)
            svg.rect(x, y, bar_w * 0.8, bottom - y, METHOD_COLORS[method], radius=3)
            err_top = y_scale(mean_acc + std_acc, ymin, ymax, top, bottom)
            err_bot = y_scale(mean_acc - std_acc, ymin, ymax, top, bottom)
            err_x = x + bar_w * 0.4
            svg.line(err_x, err_top, err_x, err_bot, "#273444", 1.2)
            svg.line(err_x - 4, err_top, err_x + 4, err_top, "#273444", 1.2)
            svg.line(err_x - 4, err_bot, err_x + 4, err_bot, "#273444", 1.2)
        svg.text(cx, bottom + 28, setting_label(model, split), "label", "middle")
    svg.text(width / 2, 675, "Use the gain heatmap and oracle-gap plot for small accuracy differences.", "subtitle", "middle")
    svg.save(output)


def diverging_color(delta: float, max_abs: float) -> str:
    if max_abs <= 0:
        return "#ffffff"
    t = min(abs(delta) / max_abs, 1.0)
    if delta >= 0:
        start = (240, 253, 250)
        end = (15, 118, 110)
    else:
        start = (255, 241, 242)
        end = (190, 18, 60)
    rgb = tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % rgb


def sequential_color(value: float, max_value: float) -> str:
    if max_value <= 0:
        return "#f8fafc"
    t = min(max(value / max_value, 0.0), 1.0)
    start = (248, 250, 252)
    end = (96, 165, 250)
    rgb = tuple(int(start[i] + (end[i] - start[i]) * t) for i in range(3))
    return "#%02x%02x%02x" % rgb


def draw_gain_heatmap(compare_rows: List[Dict[str, object]], output: Path) -> None:
    width, height = 860, 520
    svg = Svg(width, height)
    svg.text(40, 42, "FedFisher gain over FedAvg", "title")
    svg.text(40, 66, "Absolute test accuracy gain in percentage points; text shows mean gain and seed wins.", "subtitle")
    settings = sorted({(str(row["model_type"]), str(row["split"])) for row in compare_rows}, key=setting_sort_key)
    by_key = {(row["model_type"], row["split"], row["method"]): row for row in compare_rows}
    deltas = [
        float(row["gain_over_fedavg_mean_pct"])
        for row in compare_rows
        if row["method"] in FEDFISHER_METHODS
    ]
    max_abs = max(abs(delta) for delta in deltas) if deltas else 1.0
    left, top = 175, 115
    cell_w, cell_h = 185, 70
    for col_idx, method in enumerate(FEDFISHER_METHODS):
        svg.text(left + col_idx * cell_w + cell_w / 2, top - 18, METHOD_LABELS[method], "label", "middle")
    for row_idx, (model, split) in enumerate(settings):
        svg.text(left - 18, top + row_idx * cell_h + cell_h / 2 + 4, setting_label(model, split), "label", "end")
        for col_idx, method in enumerate(FEDFISHER_METHODS):
            x = left + col_idx * cell_w
            y = top + row_idx * cell_h
            row = by_key.get((model, split, method))
            if row is None:
                svg.rect(x, y, cell_w - 8, cell_h - 8, "#f1f5f9", stroke="#ffffff", radius=4)
                svg.text(x + cell_w / 2 - 4, y + 39, "n/a", "small", "middle")
                continue
            delta = float(row["gain_over_fedavg_mean_pct"])
            wins = int(row["seed_wins"])
            n = int(row["n"])
            svg.rect(x, y, cell_w - 8, cell_h - 8, diverging_color(delta, max_abs), stroke="#ffffff", radius=4)
            sign = "+" if delta >= 0 else ""
            svg.text(x + cell_w / 2 - 4, y + 30, f"{sign}{delta:.2f} pp", "label", "middle", weight="700")
            svg.text(x + cell_w / 2 - 4, y + 50, f"{wins}/{n} seed wins", "small", "middle")
    svg.text(40, height - 36, "Teal means positive gain; red means lower accuracy than one-shot FedAvg.", "subtitle")
    svg.save(output)


def draw_mlp_gain_focus(compare_rows: List[Dict[str, object]], output: Path) -> None:
    mlp_rows = [row for row in compare_rows if row["model_type"] == "mlp"]
    if not mlp_rows:
        return
    width, height = 940, 560
    left, right, top, bottom = 80, 45, 115, 430
    svg = Svg(width, height)
    svg.text(40, 42, "MLP gain over FedAvg", "title")
    svg.text(40, 66, "Pooled oracle and FedFisher variants, shown in absolute accuracy points.", "subtitle")
    methods = [method for method in ["pool", "fedfisher_diag", "fedfisher_kfac"] if any(row["method"] == method for row in mlp_rows)]
    draw_legend(svg, 420, 52, methods, step=150)

    splits = [split for split in SPLIT_ORDER if any(row["split"] == split for row in mlp_rows)]
    by_key = {(row["split"], row["method"]): row for row in mlp_rows}
    values = []
    for row in mlp_rows:
        if row["method"] in methods:
            values.append(float(row["gain_over_fedavg_mean_pct"]) + float(row["gain_over_fedavg_std_pct"]))
            values.append(float(row["gain_over_fedavg_mean_pct"]) - float(row["gain_over_fedavg_std_pct"]))
    ymin = math.floor((min(values + [0.0]) - 0.25) * 2.0) / 2.0
    ymax = math.ceil((max(values + [0.0]) + 0.25) * 2.0) / 2.0
    if ymax == ymin:
        ymax += 1.0
    tick = math.ceil(ymin)
    while tick <= math.floor(ymax):
        y = y_scale(float(tick), ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{tick:+d}", "axis", "end")
        tick += 1
    zero_y = y_scale(0.0, ymin, ymax, top, bottom)
    svg.line(left, zero_y, width - right, zero_y, "#273444", 1.3)
    svg.line(left, top, left, bottom, cls="axisline")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.text(22, 275, "Gain vs FedAvg (pp)", "label", rotate=-90, anchor="middle")

    group_w = (width - left - right) / len(splits)
    for split_idx, split in enumerate(splits):
        cx = left + group_w * (split_idx + 0.5)
        bar_w = min(58.0, group_w / (len(methods) + 2))
        start_x = cx - bar_w * len(methods) / 2.0
        for method_idx, method in enumerate(methods):
            row = by_key.get((split, method))
            if row is None:
                continue
            value = float(row["gain_over_fedavg_mean_pct"])
            std = float(row["gain_over_fedavg_std_pct"])
            x = start_x + method_idx * bar_w
            y = y_scale(value, ymin, ymax, top, bottom)
            bar_y = min(y, zero_y)
            bar_h = max(abs(zero_y - y), 1.0)
            svg.rect(x, bar_y, bar_w * 0.78, bar_h, METHOD_COLORS[method], radius=3)
            err_top = y_scale(value + std, ymin, ymax, top, bottom)
            err_bot = y_scale(value - std, ymin, ymax, top, bottom)
            err_x = x + bar_w * 0.39
            svg.line(err_x, err_top, err_x, err_bot, "#273444", 1.1)
            svg.line(err_x - 4, err_top, err_x + 4, err_top, "#273444", 1.1)
            svg.line(err_x - 4, err_bot, err_x + 4, err_bot, "#273444", 1.1)
            label_y = y - 8 if value >= 0 else y + 16
            svg.text(err_x, label_y, f"{value:+.2f}", "small", "middle")
        svg.text(cx, bottom + 28, SPLIT_LABELS.get(split, split), "label", "middle")
    svg.text(width / 2, 510, "Positive bars improve over one-shot FedAvg; Pool is the centralized reference.", "subtitle", "middle")
    svg.save(output)


def draw_oracle_gap_bars(compare_rows: List[Dict[str, object]], output: Path) -> None:
    rows = [row for row in compare_rows if row["method"] != "pool"]
    if not rows:
        return
    width, height = 1180, 650
    left, right, top, bottom = 90, 40, 105, 510
    svg = Svg(width, height)
    svg.text(40, 42, "Gap to pooled oracle", "title")
    svg.text(40, 66, "Mean test accuracy difference from the centralized pooled baseline.", "subtitle")
    methods = [method for method in METHOD_ORDER if method != "pool" and any(row["method"] == method for row in rows)]
    draw_legend(svg, 430, 52, methods, step=140)

    settings = sorted({(str(row["model_type"]), str(row["split"])) for row in compare_rows}, key=setting_sort_key)
    by_key = {(row["model_type"], row["split"], row["method"]): row for row in compare_rows}
    values = []
    for row in rows:
        values.append(float(row["gap_to_pool_mean_pct"]) + float(row["gap_to_pool_std_pct"]))
        values.append(float(row["gap_to_pool_mean_pct"]) - float(row["gap_to_pool_std_pct"]))
    ymin = math.floor((min(values + [0.0]) - 0.25) * 2.0) / 2.0
    ymax = math.ceil((max(values + [0.0]) + 0.25) * 2.0) / 2.0
    if ymax == ymin:
        ymax += 1.0
    tick = math.ceil(ymin)
    while tick <= math.floor(ymax):
        y = y_scale(float(tick), ymin, ymax, top, bottom)
        svg.line(left, y, width - right, y, cls="grid")
        svg.text(left - 12, y + 4, f"{tick:+d}", "axis", "end")
        tick += 1
    zero_y = y_scale(0.0, ymin, ymax, top, bottom)
    svg.line(left, zero_y, width - right, zero_y, "#273444", 1.3)
    svg.line(left, top, left, bottom, cls="axisline")
    svg.line(left, bottom, width - right, bottom, cls="axisline")
    svg.text(24, 310, "Gap to Pool (pp)", "label", rotate=-90, anchor="middle")

    group_w = (width - left - right) / len(settings)
    for idx, (model, split) in enumerate(settings):
        present = [method for method in methods if (model, split, method) in by_key]
        cx = left + group_w * (idx + 0.5)
        bar_w = min(34.0, group_w / (len(present) + 2))
        start_x = cx - bar_w * len(present) / 2.0
        for method_idx, method in enumerate(present):
            row = by_key[(model, split, method)]
            value = float(row["gap_to_pool_mean_pct"])
            std = float(row["gap_to_pool_std_pct"])
            x = start_x + method_idx * bar_w
            y = y_scale(value, ymin, ymax, top, bottom)
            bar_y = min(y, zero_y)
            bar_h = max(abs(zero_y - y), 1.0)
            svg.rect(x, bar_y, bar_w * 0.8, bar_h, METHOD_COLORS[method], radius=3)
            err_top = y_scale(value + std, ymin, ymax, top, bottom)
            err_bot = y_scale(value - std, ymin, ymax, top, bottom)
            err_x = x + bar_w * 0.4
            svg.line(err_x, err_top, err_x, err_bot, "#273444", 1.1)
            svg.line(err_x - 3.5, err_top, err_x + 3.5, err_top, "#273444", 1.1)
            svg.line(err_x - 3.5, err_bot, err_x + 3.5, err_bot, "#273444", 1.1)
        svg.text(cx, bottom + 28, setting_label(model, split), "label", "middle")
    svg.text(width / 2, 610, "Values above zero are possible on individual finite samples but should stay small for a strong pooled baseline.", "subtitle", "middle")
    svg.save(output)


def draw_fisher_selection(compare_rows: List[Dict[str, object]], output: Path) -> None:
    settings = sorted({(str(row["model_type"]), str(row["split"])) for row in compare_rows}, key=setting_sort_key)
    methods = [
        method
        for method in ["fedfisher_diag", "fedfisher_kfac"]
        if any(row["method"] == method and row["fisher_val_score_mean_pct"] != "" for row in compare_rows)
    ]
    if not methods:
        return
    width = 260 + 235 * len(methods)
    height = 180 + 86 * len(settings)
    svg = Svg(width, height)
    svg.text(40, 42, "FedFisher server selection", "title")
    svg.text(40, 66, "Validation-selected Adam step after starting from FedAvg.", "subtitle")
    by_key = {(row["model_type"], row["split"], row["method"]): row for row in compare_rows}
    max_step = max(
        float(row["selected_step_mean"])
        for row in compare_rows
        if row["method"] in methods and row["fisher_val_score_mean_pct"] != ""
    )
    left, top = 190, 120
    cell_w, cell_h = 235, 74
    for col_idx, method in enumerate(methods):
        svg.text(left + col_idx * cell_w + cell_w / 2, top - 18, METHOD_LABELS[method], "label", "middle")
    for row_idx, (model, split) in enumerate(settings):
        y = top + row_idx * cell_h
        svg.text(left - 16, y + cell_h / 2 + 4, setting_label(model, split), "label", "end")
        for col_idx, method in enumerate(methods):
            x = left + col_idx * cell_w
            row = by_key.get((model, split, method))
            if row is None or row["fisher_val_score_mean_pct"] == "":
                svg.rect(x, y, cell_w - 10, cell_h - 10, "#f1f5f9", stroke="#ffffff", radius=4)
                svg.text(x + cell_w / 2 - 5, y + 39, "n/a", "small", "middle")
                continue
            step_mean = float(row["selected_step_mean"])
            moved = int(row["selected_step_nonzero"])
            n = int(row["n"])
            val_score = float(row["fisher_val_score_mean_pct"])
            svg.rect(x, y, cell_w - 10, cell_h - 10, sequential_color(step_mean, max_step), stroke="#ffffff", radius=4)
            svg.text(x + cell_w / 2 - 5, y + 28, f"step {step_mean:.0f}", "label", "middle", weight="700")
            svg.text(x + cell_w / 2 - 5, y + 47, f"{moved}/{n} moved, val {val_score:.1f}%", "small", "middle")
    svg.text(40, height - 34, "Step 0 means validation kept the FedAvg initialization.", "subtitle")
    svg.save(output)


def draw_split_lines(compare_rows: List[Dict[str, object]], output: Path) -> None:
    width, height = 1040, 620
    svg = Svg(width, height)
    svg.text(40, 42, "Accuracy across client splits", "title")
    svg.text(40, 66, "Lines compare IID and non-IID splits for each model family.", "subtitle")
    draw_legend(svg, 410, 52, METHOD_ORDER, step=122)
    by_key = {(row["model_type"], row["split"], row["method"]): row for row in compare_rows}
    acc_values = [float(row["accuracy_mean_pct"]) for row in compare_rows if row["method"] != "pool"]
    ymin, ymax = accuracy_axis(acc_values)
    panel_w, panel_h = 410, 380
    start_x, start_y = 95, 130
    gap = 80
    x_splits = [split for split in SPLIT_ORDER if any((model, split, method) in by_key for model in MODEL_ORDER for method in METHOD_ORDER)]

    for model_idx, model in enumerate([model for model in MODEL_ORDER if any(row["model_type"] == model for row in compare_rows)]):
        x0 = start_x + model_idx * (panel_w + gap)
        y0 = start_y
        svg.text(x0, y0 - 18, MODEL_LABELS.get(model, model), "label", weight="700")
        for tick in range(int(math.ceil(ymin / 5.0) * 5), int(math.floor(ymax / 5.0) * 5) + 1, 5):
            y = y_scale(tick, ymin, ymax, y0, y0 + panel_h)
            svg.line(x0, y, x0 + panel_w, y, cls="grid")
            svg.text(x0 - 8, y + 4, tick, "small", "end")
        svg.line(x0, y0, x0, y0 + panel_h, cls="axisline")
        svg.line(x0, y0 + panel_h, x0 + panel_w, y0 + panel_h, cls="axisline")
        xs = {}
        for split_idx, split in enumerate(x_splits):
            x = x0 + 60 + split_idx * ((panel_w - 120) / max(len(x_splits) - 1, 1))
            xs[split] = x
            svg.text(x, y0 + panel_h + 25, SPLIT_LABELS.get(split, split), "label", "middle")
        for method in METHOD_ORDER:
            points = []
            for split in x_splits:
                row = by_key.get((model, split, method))
                if row is not None:
                    points.append((xs[split], y_scale(float(row["accuracy_mean_pct"]), ymin, ymax, y0, y0 + panel_h)))
            if len(points) >= 2:
                svg.path(points, METHOD_COLORS[method], 2.4)
            for x, y in points:
                svg.circle(x, y, 4, METHOD_COLORS[method], "#ffffff", 1.3)
    svg.text(26, 320, "Test accuracy (%)", "label", rotate=-90, anchor="middle")
    svg.save(output)


def draw_seed_pairs(rows: List[Dict[str, str]], output: Path) -> None:
    width, height = 1180, 760
    svg = Svg(width, height)
    svg.text(40, 42, "Seed-level paired comparison", "title")
    svg.text(40, 66, "Each line connects one seed's FedAvg accuracy to a FedFisher variant.", "subtitle")
    draw_legend(svg, 590, 52, ["fedavg_oneshot", "fedfisher_diag", "fedfisher_full", "fedfisher_kfac"], step=140)
    by_seed = {
        (row["model_type"], row["split"], row["method"], int(row["seed"])): float(row["accuracy"]) * 100.0
        for row in rows
    }
    settings = sorted({(row["model_type"], row["split"]) for row in rows}, key=setting_sort_key)
    all_values = [float(row["accuracy"]) * 100.0 for row in rows if row["method"] != "pool"]
    ymin, ymax = accuracy_axis(all_values)
    panel_w, panel_h = 505, 220
    start_x, start_y = 85, 125
    gap_x, gap_y = 45, 70

    for idx, (model, split) in enumerate(settings):
        col = idx % 2
        row_idx = idx // 2
        x0 = start_x + col * (panel_w + gap_x)
        y0 = start_y + row_idx * (panel_h + gap_y)
        svg.text(x0, y0 - 14, setting_label(model, split), "label", weight="700")
        for tick in range(int(math.ceil(ymin / 5.0) * 5), int(math.floor(ymax / 5.0) * 5) + 1, 5):
            y = y_scale(tick, ymin, ymax, y0, y0 + panel_h - 38)
            svg.line(x0, y, x0 + panel_w, y, cls="grid")
            svg.text(x0 - 8, y + 4, tick, "small", "end")
        svg.line(x0, y0, x0, y0 + panel_h - 38, cls="axisline")
        svg.line(x0, y0 + panel_h - 38, x0 + panel_w, y0 + panel_h - 38, cls="axisline")

        present = ["fedavg_oneshot"] + [
            method
            for method in FEDFISHER_METHODS
            if any((model, split, method, seed) in by_seed for seed in range(100))
        ]
        x_positions = {
            method: x0 + 50 + idx_method * ((panel_w - 100) / max(len(present) - 1, 1))
            for idx_method, method in enumerate(present)
        }
        for method in present:
            svg.text(x_positions[method], y0 + panel_h - 12, METHOD_LABELS[method].replace("FedFisher ", ""), "small", "middle")
        seeds = sorted({seed for (m, s, _method, seed) in by_seed if m == model and s == split})
        offsets = [-5, -2.5, 0, 2.5, 5]
        for seed_idx, seed in enumerate(seeds):
            fedavg = by_seed[(model, split, "fedavg_oneshot", seed)]
            y_fed = y_scale(fedavg, ymin, ymax, y0, y0 + panel_h - 38)
            x_fed = x_positions["fedavg_oneshot"] + offsets[seed_idx % len(offsets)] * 0.25
            svg.circle(x_fed, y_fed, 3.3, METHOD_COLORS["fedavg_oneshot"], "#ffffff", 1)
            for method in present[1:]:
                method_key = (model, split, method, seed)
                if method_key not in by_seed:
                    continue
                value = by_seed[method_key]
                x_method = x_positions[method] + offsets[seed_idx % len(offsets)] * 0.25
                y_method = y_scale(value, ymin, ymax, y0, y0 + panel_h - 38)
                svg.line(x_fed, y_fed, x_method, y_method, "#cbd2d9", 1.1)
                svg.circle(x_method, y_method, 3.3, METHOD_COLORS[method], "#ffffff", 1)
    svg.text(24, 390, "Test accuracy (%)", "label", rotate=-90, anchor="middle")
    svg.text(width / 2, 725, "Upward sloping lines indicate the FedFisher variant improves over FedAvg for that seed.", "subtitle", "middle")
    svg.save(output)


def write_readme(compare_rows: List[Dict[str, object]], output: Path, source: Path) -> None:
    fisher_rows = [row for row in compare_rows if row["method"] in FEDFISHER_METHODS]
    best = max(fisher_rows, key=lambda row: float(row["gain_over_fedavg_mean_pct"]))
    worst = min(fisher_rows, key=lambda row: float(row["gain_over_fedavg_mean_pct"]))
    mlp_noniid_rows = [
        row
        for row in fisher_rows
        if row["model_type"] == "mlp" and row["split"] == "noniid"
    ]
    mlp_noniid_best = max(mlp_noniid_rows, key=lambda row: float(row["gain_over_fedavg_mean_pct"])) if mlp_noniid_rows else None
    lines = [
        "# Synthetic FedFisher Visualizations",
        "",
        f"Generated from `{source}`.",
        "",
        "Files:",
        "",
        "- `synthetic_compare_summary.csv`: numeric comparison table in accuracy percentage points.",
        "- `synthetic_accuracy_bars.svg`: absolute mean accuracy bars with std error bars.",
        "- `synthetic_gain_heatmap.svg`: FedFisher gain over one-shot FedAvg.",
        "- `synthetic_mlp_gain_focus.svg`: MLP-only gain over FedAvg, including the pooled oracle.",
        "- `synthetic_oracle_gap.svg`: federated method gap to the pooled oracle.",
        "- `synthetic_fisher_selection.svg`: validation-selected FedFisher server step.",
        "- `synthetic_split_lines.svg`: accuracy trend from IID to non-IID splits.",
        "- `synthetic_seed_pairs.svg`: paired seed-level comparison.",
        "",
        "Largest FedFisher gain: %s %s %s, %+0.2f accuracy points."
        % (
            MODEL_LABELS.get(str(best["model_type"]), str(best["model_type"])),
            SPLIT_LABELS.get(str(best["split"]), str(best["split"])),
            METHOD_LABELS[str(best["method"])],
            float(best["gain_over_fedavg_mean_pct"]),
        ),
        "Smallest FedFisher gain: %s %s %s, %+0.2f accuracy points."
        % (
            MODEL_LABELS.get(str(worst["model_type"]), str(worst["model_type"])),
            SPLIT_LABELS.get(str(worst["split"]), str(worst["split"])),
            METHOD_LABELS[str(worst["method"])],
            float(worst["gain_over_fedavg_mean_pct"]),
        ),
    ]
    if mlp_noniid_best is not None:
        lines.append(
            "Best MLP non-IID FedFisher gain: %s, %+0.2f accuracy points."
            % (
                METHOD_LABELS[str(mlp_noniid_best["method"])],
                float(mlp_noniid_best["gain_over_fedavg_mean_pct"]),
            )
        )
    lines.append("Interpretation: FedFisher improves over FedAvg in MLP non-IID here, but not across every model/split.")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_results(input_path)
    compare_rows = build_compare_rows(rows)
    write_compare_csv(compare_rows, output_dir / "synthetic_compare_summary.csv")
    draw_accuracy_bars(compare_rows, output_dir / "synthetic_accuracy_bars.svg")
    draw_gain_heatmap(compare_rows, output_dir / "synthetic_gain_heatmap.svg")
    draw_mlp_gain_focus(compare_rows, output_dir / "synthetic_mlp_gain_focus.svg")
    draw_oracle_gap_bars(compare_rows, output_dir / "synthetic_oracle_gap.svg")
    draw_fisher_selection(compare_rows, output_dir / "synthetic_fisher_selection.svg")
    draw_split_lines(compare_rows, output_dir / "synthetic_split_lines.svg")
    draw_seed_pairs(rows, output_dir / "synthetic_seed_pairs.svg")
    write_readme(compare_rows, output_dir / "README.md", input_path)
    print(f"Wrote synthetic comparison figures to {output_dir}")


if __name__ == "__main__":
    main()
