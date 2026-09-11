#!/usr/bin/env python3
"""Compare one-shot LASSO with historical FedAvg and FedFisher results.

The comparison uses the shared SyntheticEffectModifier primary endpoint:
``permute / tau_intervention_logit``.  It deliberately keeps the linear
logistic LASSO reference distinct from the shallow and deep MLP methods.
"""

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


DATA_ROOT = Path("/data/del6500/FedFisher")
OUTPUTS_ROOT = DATA_ROOT / "effect_modifier_experiment" / "outputs"
RESULTS_ROOT = Path(
    os.environ.get(
        "ONE_SHOT_COMPARISON_RESULTS_DIR",
        DATA_ROOT / "results" / "one_shot_lasso_vs_fedfisher",
    )
)

VARIANTS = [
    "control",
    "local_permutation",
    "site_gaussian_shift",
    "site_gaussian_local_permutation",
]
VARIANT_LABELS = {
    "control": "Control",
    "local_permutation": "Local perm.",
    "site_gaussian_shift": "Gaussian shift",
    "site_gaussian_local_permutation": "Combined",
}

OLD_RUNS = {
    "control": "prediction_intervention_factorial_90700c2_300seeds",
    "local_permutation": "local_permutation_factorial_90700c2_300seeds",
    "site_gaussian_shift": "site_gaussian_shift_factorial_90700c2_300seeds",
    "site_gaussian_local_permutation": (
        "site_gaussian_local_permutation_factorial_90700c2_300seeds"
    ),
}
NEW_FAMILIES = [
    "one_shot_local_logistic_lasso_fedavg",
    "one_shot_mlp_group_lasso_fedavg",
]
NEW_RUN_SUFFIX = "stationarity_4000iter_full_scope_server_scheduler_v2r2_300seeds"

PLOT_METHODS = [
    "logistic_lasso",
    "fedavg",
    "fedfisher_diag",
    "fedfisher_kfac",
]
METHOD_LABELS = {
    "logistic_lasso": "Logistic LASSO + FedAvg",
    "mlp_group_lasso": "MLP group-LASSO + FedAvg",
    "fedavg": "MLP FedAvg",
    "fedfisher_diag": "MLP FedFisher diag",
    "fedfisher_kfac": "MLP FedFisher KFAC",
}
METHOD_COLORS = {
    "logistic_lasso": "#D55E00",
    "mlp_group_lasso": "#CC79A7",
    "fedavg": "#737373",
    "fedfisher_diag": "#0072B2",
    "fedfisher_kfac": "#009E73",
}
PANELS = [
    ("SyntheticMLP", "Shallow MLP comparison"),
    ("SyntheticMLPDeep", "Deep MLP comparison"),
]

COMMON_CONFIG = {
    "seed_start": "0",
    "seed_end": "299",
    "synthetic_alpha": "0.1",
    "effect_modifier_num_sites": "10",
    "synthetic_num_test": "10000",
    "effect_modifier_covariate_dim": "100",
    "effect_modifier_signal_dim": "2",
    "effect_modifier_signal_strength": "2.0",
    "effect_modifier_main_effect_strength": "1.0",
    "effect_modifier_confounding_strength": "1.0",
    "effect_modifier_intercept": "0.0",
    "effect_modifier_treatment_prob": "0.5",
    "effect_modifier_propensity_clip": "0.05",
    "effect_modifier_site_min_samples": "500",
    "effect_modifier_site_max_samples": "1000",
    "prediction_intervention_repeats": "5",
    "prediction_intervention_batch_size": "1024",
}


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def std_error(values):
    return sample_std(values) / math.sqrt(len(values))


def parse_manifest(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def manifest_config(manifest):
    if "run_config_json" not in manifest:
        return manifest
    config = json.loads(manifest["run_config_json"])
    return {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in config.items()}


def resolve_inside(path, root, label, strict):
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=strict)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label} resolves outside {resolved_root}: {resolved}")
    return resolved


def validate_run(run_id, variant):
    run_dir = resolve_inside(OUTPUTS_ROOT / run_id, OUTPUTS_ROOT, "input run", True)
    manifest_path = run_dir / "run_manifest.env"
    manifest = parse_manifest(manifest_path)
    if manifest.get("format_version") != "2":
        raise ValueError(f"{run_id}: expected format_version=2")
    if manifest.get("run_id") != run_id:
        raise ValueError(f"{run_id}: manifest run_id mismatch")
    if manifest.get("effect_modifier_variant") != variant:
        raise ValueError(f"{run_id}: variant mismatch")
    config = manifest_config(manifest)
    for key, expected in COMMON_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"{run_id}: {key}={actual!r}, expected {expected!r}")
    return run_dir, manifest


def ensure_output_tree():
    resolved = resolve_inside(RESULTS_ROOT, DATA_ROOT / "results", "results directory", False)
    resolved.mkdir(parents=True, exist_ok=True)
    resolved = resolve_inside(resolved, DATA_ROOT / "results", "results directory", True)
    unsafe = next((path for path in resolved.rglob("*") if path.is_symlink()), None)
    if unsafe is not None:
        raise ValueError(f"results tree contains a symlink: {unsafe}")
    return resolved


def method_key(model_source):
    mapping = {
        "one_shot_local_logistic_lasso_fedavg": "logistic_lasso",
        "one_shot_mlp_group_lasso_fedavg": "mlp_group_lasso",
        "fedavg": "fedavg",
        "fedfisher_diag": "fedfisher_diag",
        "fedfisher_kfac": "fedfisher_kfac",
    }
    return mapping.get(model_source)


def collect_run(run_dir, manifest, variant, allowed_sources, buckets):
    summary_paths = sorted(run_dir.glob("*_prediction_intervention_summary.csv"))
    model_paths = sorted(run_dir.glob("*_prediction_intervention_model_summary.csv"))
    if not summary_paths or not model_paths:
        raise ValueError(f"{run_dir.name}: missing result files")

    seen_primary = set()
    for path in summary_paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model_source"] not in allowed_sources:
                    continue
                if row["intervention_mode"] != "permute":
                    continue
                if row["metric"] != "tau_intervention_logit":
                    continue
                method = method_key(row["model_source"])
                key = (variant, row["split"], row["model"], method)
                identity = key + (int(row["seed"]),)
                if identity in seen_primary:
                    raise ValueError(f"duplicate primary result: {identity}")
                seen_primary.add(identity)
                bucket = buckets[key]
                bucket["exact_top2"].append(float(row["topk_hits"]) == 2.0)
                bucket["auroc"].append(float(row["auroc"]))
                bucket["seeds"].add(int(row["seed"]))
                bucket["run_id"] = manifest["run_id"]
                bucket["git_commit"] = manifest["git_commit"]

    seen_model = set()
    for path in model_paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["model_source"] not in allowed_sources:
                    continue
                method = method_key(row["model_source"])
                key = (variant, row["split"], row["model"], method)
                identity = key + (int(row["seed"]),)
                if identity in seen_model:
                    raise ValueError(f"duplicate model result: {identity}")
                seen_model.add(identity)
                buckets[key]["accuracy"].append(float(row["baseline_accuracy"]))


def collect_results():
    buckets = defaultdict(
        lambda: {
            "exact_top2": [],
            "auroc": [],
            "accuracy": [],
            "seeds": set(),
            "run_id": "",
            "git_commit": "",
        }
    )
    for variant, run_id in OLD_RUNS.items():
        run_dir, manifest = validate_run(run_id, variant)
        collect_run(
            run_dir,
            manifest,
            variant,
            {"fedavg", "fedfisher_diag", "fedfisher_kfac"},
            buckets,
        )
    for family in NEW_FAMILIES:
        for variant in VARIANTS:
            run_id = f"{family}_{variant}_{NEW_RUN_SUFFIX}"
            run_dir, manifest = validate_run(run_id, variant)
            collect_run(run_dir, manifest, variant, {family}, buckets)

    rows = []
    for (variant, split, model, method), values in sorted(buckets.items()):
        counts = {
            len(values["exact_top2"]),
            len(values["auroc"]),
            len(values["accuracy"]),
            len(values["seeds"]),
        }
        if counts != {300} or values["seeds"] != set(range(300)):
            raise ValueError(
                f"incomplete group {(variant, split, model, method)}: counts={counts}"
            )
        exact = values["exact_top2"]
        exact_prop = mean(exact)
        rows.append(
            {
                "variant": variant,
                "split": split,
                "model": model,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": 300,
                "exact_top2_count": int(sum(exact)),
                "exact_top2_pct": exact_prop * 100.0,
                "exact_top2_se_pct": math.sqrt(exact_prop * (1.0 - exact_prop) / 300) * 100.0,
                "auroc_mean": mean(values["auroc"]),
                "auroc_se": std_error(values["auroc"]),
                "accuracy_pct_mean": mean(values["accuracy"]) * 100.0,
                "accuracy_pct_se": std_error(values["accuracy"]) * 100.0,
                "run_id": values["run_id"],
                "git_commit": values["git_commit"],
            }
        )
    if len(rows) != 72:
        raise ValueError(f"expected 72 aggregate rows, found {len(rows)}")
    return rows


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rendered = {
                key: f"{value:.8f}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(rendered)


def esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(body, x, y, value, cls="", anchor="start", rotate=None):
    attrs = f' class="{cls}"' if cls else ""
    attrs += f' text-anchor="{anchor}"'
    if rotate is not None:
        attrs += f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    body.append(f'<text x="{x:.1f}" y="{y:.1f}"{attrs}>{esc(value)}</text>')


def line(body, x1, y1, x2, y2, cls="", stroke=None, width=1):
    attrs = f' class="{cls}"' if cls else ""
    if stroke:
        attrs += f' stroke="{stroke}" stroke-width="{width}"'
    body.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"{attrs}/>')


def svg_document(width, height, body):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="{width}" height="{height}" fill="#ffffff"/>
<style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#111827}}
.title{{font-size:24px;font-weight:700}}
.subtitle{{font-size:13px;fill:#4b5563}}
.panel{{font-size:16px;font-weight:700}}
.axis{{font-size:12px;fill:#374151}}
.tick{{font-size:11px;fill:#4b5563}}
.legend{{font-size:12px}}
.value{{font-size:9px;font-weight:600}}
.grid{{stroke:#e5e7eb;stroke-width:1}}
.frame{{stroke:#9ca3af;stroke-width:1;fill:none}}
</style>
{''.join(body)}
</svg>
'''


def draw_legend(body, y=84):
    widths = {
        "logistic_lasso": 188,
        "fedavg": 105,
        "fedfisher_diag": 165,
        "fedfisher_kfac": 175,
    }
    total = sum(widths[method] for method in PLOT_METHODS)
    cursor = (1520 - total) / 2
    for method in PLOT_METHODS:
        item_width = widths[method]
        body.append(
            f'<rect x="{cursor:.1f}" y="{y - 11:.1f}" width="13" height="13" '
            f'fill="{METHOD_COLORS[method]}"/>'
        )
        text(body, cursor + 19, y, METHOD_LABELS[method], "legend")
        cursor += item_width


def row_lookup(rows):
    return {
        (row["variant"], row["split"], row["model"], row["method"]): row
        for row in rows
    }


def comparison_row(lookup, variant, split, panel_model, method):
    if method == "logistic_lasso":
        model = "linear"
    else:
        model = panel_model
    return lookup[(variant, split, model, method)]


def draw_grouped_chart(
    rows,
    output_path,
    title_value,
    subtitle,
    groups,
    metric,
    error,
    y_min,
    y_max,
    ticks,
    y_label,
    value_format=None,
):
    lookup = row_lookup(rows)
    width, height = 1520, 760
    body = []
    text(body, 42, 38, title_value, "title")
    text(body, 42, 62, subtitle, "subtitle")
    draw_legend(body)
    panel_top, panel_bottom = 142, 628
    panel_width = 640
    panel_lefts = [92, 824]

    def y_scale(value):
        value = max(y_min, min(y_max, value))
        return panel_bottom - (value - y_min) / (y_max - y_min) * (panel_bottom - panel_top)

    for panel_index, ((panel_model, panel_label), panel_left) in enumerate(zip(PANELS, panel_lefts)):
        text(body, panel_left - 44, 122, chr(ord("A") + panel_index), "panel")
        text(body, panel_left + panel_width / 2, 122, panel_label, "panel", "middle")
        for tick in ticks:
            yy = y_scale(tick)
            line(body, panel_left, yy, panel_left + panel_width, yy, "grid")
            text(body, panel_left - 9, yy + 4, f"{tick:g}", "tick", "end")
        body.append(
            f'<rect x="{panel_left}" y="{panel_top}" width="{panel_width}" '
            f'height="{panel_bottom - panel_top}" class="frame"/>'
        )
        text(
            body,
            panel_left - 60,
            (panel_top + panel_bottom) / 2,
            y_label,
            "axis",
            "middle",
            -90,
        )
        group_width = panel_width / len(groups)
        bar_width = min(26.0, group_width / (len(PLOT_METHODS) + 2.0))
        method_center = (len(PLOT_METHODS) - 1) / 2.0
        for group_index, (variant, split, group_label) in enumerate(groups):
            center = panel_left + group_width * (group_index + 0.5)
            for method_index, method in enumerate(PLOT_METHODS):
                row = comparison_row(lookup, variant, split, panel_model, method)
                value = float(row[metric])
                se = float(row[error])
                x = center + (method_index - method_center) * (bar_width + 4)
                yy = y_scale(value)
                base = y_scale(y_min)
                body.append(
                    f'<rect x="{x - bar_width / 2:.1f}" y="{yy:.1f}" '
                    f'width="{bar_width:.1f}" height="{max(0.0, base - yy):.1f}" '
                    f'fill="{METHOD_COLORS[method]}"/>'
                )
                high = y_scale(value + se)
                low = y_scale(value - se)
                line(body, x, high, x, low, stroke="#111827")
                line(body, x - 3, high, x + 3, high, stroke="#111827")
                line(body, x - 3, low, x + 3, low, stroke="#111827")
                if value_format:
                    text(body, x + 2, yy - 5, value_format.format(value), "value", "start", -65)
            text(body, center, panel_bottom + 24, group_label, "axis", "middle")

    text(body, 42, 690, "Bars show means across 300 seeds; whiskers are standard errors.", "subtitle")
    text(
        body,
        42,
        712,
        "The linear logistic LASSO reference is repeated in both panels; all other bars use the panel's MLP architecture.",
        "subtitle",
    )
    output_path.write_text(svg_document(width, height, body))


def write_readme(path):
    path.write_text(
        """# One-shot LASSO, FedAvg, and FedFisher comparison

These figures combine the 300-seed protocol-v2r2 one-shot LASSO runs with the
300-seed historical FedAvg/FedFisher factorial runs. The data generator,
prediction-intervention evaluator, FedAvg implementation, and Fisher
aggregation implementation are unchanged between the two recorded commits.

Source epochs:

- FedAvg/FedFisher: `90700c2afcaeb7ce8f3455f49d493b06655d73a7`
- One-shot LASSO: `cbcde8fdf7998aeed6601c6d1ecd0fde3915dffe`

Figures:

- `noniid_exact_top2_recovery.svg`: exact recovery of both true effect modifiers.
- `noniid_tau_intervention_auroc.svg`: primary feature-ranking AUROC.
- `heldout_prediction_accuracy.svg`: IID and non-IID prediction accuracy under
  control and site-Gaussian training distributions.
- Matching `.png` files are presentation-ready raster copies.
- `comparison_summary.csv` retains all 72 aggregated method/model/condition
  rows, including the MLP group-LASSO rows, for auditability.

The primary recovery endpoint is `permute / tau_intervention_logit`. Error bars
are standard errors over 300 seeds. Logistic LASSO is a linear model and is
shown as a shared system-level reference in both MLP panels. The figures omit
MLP group-LASSO because sparsity of its explicit first-layer interaction-input
columns is not an identifiable effect-modifier selection target in an MLP.
"""
    )


def main():
    results_dir = ensure_output_tree()
    rows = collect_results()
    write_csv(results_dir / "comparison_summary.csv", rows)
    recovery_groups = [
        (variant, "noniid", VARIANT_LABELS[variant]) for variant in VARIANTS
    ]
    draw_grouped_chart(
        rows,
        results_dir / "noniid_exact_top2_recovery.svg",
        "Non-IID exact Top-2 effect-modifier recovery",
        "Primary endpoint: permute / tau-intervention logit; both X1 and X2 must occupy the Top-2.",
        recovery_groups,
        "exact_top2_pct",
        "exact_top2_se_pct",
        0,
        100,
        [0, 20, 40, 60, 80, 100],
        "Seeds recovering both signals (%)",
        "{:.0f}",
    )
    draw_grouped_chart(
        rows,
        results_dir / "noniid_tau_intervention_auroc.svg",
        "Non-IID effect-modifier ranking AUROC",
        "Primary endpoint: permute / tau-intervention logit across 100 raw covariates.",
        recovery_groups,
        "auroc_mean",
        "auroc_se",
        0.70,
        1.00,
        [0.70, 0.80, 0.90, 1.00],
        "AUROC",
        None,
    )
    accuracy_groups = [
        ("control", "iid", "IID / control"),
        ("site_gaussian_shift", "iid", "IID / Gaussian"),
        ("control", "noniid", "Non-IID / control"),
        ("site_gaussian_shift", "noniid", "Non-IID / Gaussian"),
    ]
    draw_grouped_chart(
        rows,
        results_dir / "heldout_prediction_accuracy.svg",
        "Held-out prediction accuracy",
        "Control and local-permutation share training; Gaussian and combined share training, so duplicate accuracies are omitted.",
        accuracy_groups,
        "accuracy_pct_mean",
        "accuracy_pct_se",
        50,
        75,
        [50, 55, 60, 65, 70, 75],
        "Accuracy (%)",
        "{:.1f}",
    )
    write_readme(results_dir / "README.md")
    print(f"Wrote comparison figures and summary to {results_dir}")


if __name__ == "__main__":
    main()
