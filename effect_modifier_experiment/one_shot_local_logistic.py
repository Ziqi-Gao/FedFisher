"""Separation-safe site-local logistic LASSO fitting.

The SyntheticEffectModifier non-IID contract can assign only one outcome
class to a site.  With an unpenalized intercept and raw main-effect block, the
hard-label logistic likelihood then has no finite minimizer.  This module uses
a deterministic, vanishing site-local soft-target correction while leaving
the coefficient penalty mask unchanged: for a site with ``n`` rows,
``epsilon = 1 / (n + 1)`` and
``y_star = (1 - epsilon) * y + epsilon / 2``.

Fold assignment is computed from the original binary labels.  The same
site-level epsilon is used in every training and validation fold, and all
standardization, lambda construction, fitting, and validation remain local.
"""

from typing import Dict, List, Tuple

import numpy as np

from effect_modifier_experiment.dslassocov_benchmark import (
    OptimizerConfig,
    RawSite,
    aggregate_loss_and_gradient,
    aggregate_standardization,
    backtransform_coefficients,
    deterministic_site_folds,
    proximal_gradient_fit,
    standardize_sites,
)


SMOOTHING_RULE = "site_local_epsilon_1_over_n_plus_1_soft_binary_targets"


def _soft_targets(labels: np.ndarray, epsilon: float) -> np.ndarray:
    return (1.0 - epsilon) * np.asarray(labels, dtype=np.float64) + 0.5 * epsilon


def _raw_subset(
    raw_site: RawSite,
    soft_labels: np.ndarray,
    mask: np.ndarray,
) -> RawSite:
    return RawSite(
        features=np.asarray(raw_site.features[mask], dtype=np.float64, order="C"),
        labels=np.asarray(soft_labels[mask], dtype=np.float64, order="C"),
        site_idx=raw_site.site_idx,
    )


def _unpenalized_start(
    standardized_sites,
    penalty_start: int,
    optimizer: OptimizerConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    reduced_sites = []
    site_type = type(standardized_sites[0])
    for site in standardized_sites:
        reduced_sites.append(
            site_type(
                design=site.design[:, :penalty_start],
                labels=site.labels,
                site_idx=site.site_idx,
            )
        )
    reduced_coefficients, report = proximal_gradient_fit(
        reduced_sites,
        penalty_start=penalty_start,
        lambda_value=0.0,
        config=optimizer,
    )
    if not report["converged"]:
        raise RuntimeError(
            "soft-target unpenalized local start failed stationarity"
        )
    coefficients = np.zeros(standardized_sites[0].design.shape[1], dtype=np.float64)
    coefficients[:penalty_start] = reduced_coefficients
    return coefficients, report


def _fit_path(
    standardized_sites,
    penalty_start: int,
    lambda_values: np.ndarray,
    initial_coefficients: np.ndarray,
    optimizer: OptimizerConfig,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    coefficients = initial_coefficients.copy()
    path_coefficients = []
    reports = []
    for lambda_value in lambda_values:
        coefficients, report = proximal_gradient_fit(
            standardized_sites,
            penalty_start=penalty_start,
            lambda_value=float(lambda_value),
            config=optimizer,
            initial_coefficients=coefficients,
        )
        report = dict(report)
        report["lambda"] = float(lambda_value)
        report["num_nonzero_penalized"] = int(
            np.count_nonzero(coefficients[penalty_start:])
        )
        path_coefficients.append(coefficients.copy())
        reports.append(report)
    return path_coefficients, reports


def separation_safe_local_logistic_cv(
    raw_site: RawSite,
    covariate_dim: int,
    num_folds: int,
    lambda_count: int,
    lambda_min_ratio: float,
    cv_seed: int,
    scale_floor: float,
    optimizer: OptimizerConfig,
) -> Dict[str, object]:
    """Fit one site's deterministic soft-target logistic LASSO path and CV."""
    expected_feature_dim = 1 + 2 * int(covariate_dim)
    if raw_site.features.shape[1] != expected_feature_dim:
        raise ValueError("local logistic feature dimension is inconsistent")
    unique_labels = set(np.unique(raw_site.labels).tolist())
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError("fold-stratification labels must be binary")
    if lambda_count < 2 or not 0.0 < lambda_min_ratio < 1.0:
        raise ValueError("invalid local lambda path")

    num_examples = int(raw_site.features.shape[0])
    epsilon = 1.0 / float(num_examples + 1)
    soft_labels = _soft_targets(raw_site.labels, epsilon)
    assignments = deterministic_site_folds([raw_site], num_folds, cv_seed)[0]
    penalty_start = 2 + int(covariate_dim)

    full_site = RawSite(
        features=raw_site.features,
        labels=soft_labels,
        site_idx=raw_site.site_idx,
    )
    full_means, full_scales, full_standardization = aggregate_standardization(
        [full_site], scale_floor=scale_floor
    )
    full_standardized = standardize_sites([full_site], full_means, full_scales)
    full_start, unpenalized_report = _unpenalized_start(
        full_standardized, penalty_start, optimizer
    )
    _, full_gradient, _ = aggregate_loss_and_gradient(
        full_standardized, full_start
    )
    lambda_max = float(np.max(np.abs(full_gradient[penalty_start:])))
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError("could not derive a positive finite local lambda_max")
    lambda_values = np.geomspace(
        lambda_max,
        lambda_max * float(lambda_min_ratio),
        num=int(lambda_count),
        dtype=np.float64,
    )

    fold_validation_losses = np.zeros((num_folds, lambda_count), dtype=np.float64)
    fold_counts = []
    fold_reports = []
    for fold_idx in range(num_folds):
        validation_mask = assignments == fold_idx
        training_mask = ~validation_mask
        if not np.any(training_mask) or not np.any(validation_mask):
            raise ValueError("each local CV partition must contain rows")
        training_site = _raw_subset(raw_site, soft_labels, training_mask)
        validation_site = _raw_subset(raw_site, soft_labels, validation_mask)
        means, scales, standardization_report = aggregate_standardization(
            [training_site], scale_floor=scale_floor
        )
        standardized_training = standardize_sites([training_site], means, scales)
        standardized_validation = standardize_sites([validation_site], means, scales)
        start, start_report = _unpenalized_start(
            standardized_training, penalty_start, optimizer
        )
        coefficients, reports = _fit_path(
            standardized_training,
            penalty_start,
            lambda_values,
            start,
            optimizer,
        )
        for lambda_idx, candidate in enumerate(coefficients):
            loss, _, _ = aggregate_loss_and_gradient(
                standardized_validation, candidate
            )
            fold_validation_losses[fold_idx, lambda_idx] = float(loss)
        validation_count = int(np.count_nonzero(validation_mask))
        fold_counts.append(validation_count)
        fold_reports.append(
            {
                "fold_idx": int(fold_idx),
                "training_examples": int(np.count_nonzero(training_mask)),
                "validation_examples": validation_count,
                "standardization": standardization_report,
                "unpenalized_start": start_report,
                "lambda_fits": reports,
            }
        )

    aggregated_losses = np.average(
        fold_validation_losses,
        axis=0,
        weights=np.asarray(fold_counts, dtype=np.float64),
    )
    minimum_loss = float(np.min(aggregated_losses))
    tied = np.flatnonzero(
        np.isclose(aggregated_losses, minimum_loss, rtol=0.0, atol=1e-12)
    )
    selected_index = int(tied[0])

    final_coefficients, final_reports = _fit_path(
        full_standardized,
        penalty_start,
        lambda_values,
        full_start,
        optimizer,
    )
    selected_standardized = final_coefficients[selected_index]
    intercept, original_coefficients = backtransform_coefficients(
        selected_standardized, full_means, full_scales
    )
    return {
        "original_intercept": float(intercept),
        "original_coefficients": original_coefficients,
        "response_stabilization": {
            "rule": SMOOTHING_RULE,
            "epsilon": float(epsilon),
            "original_label_counts": {
                "0": int(np.count_nonzero(raw_site.labels == 0.0)),
                "1": int(np.count_nonzero(raw_site.labels == 1.0)),
            },
            "coefficient_penalty_added_to_unpenalized_block": False,
        },
        "standardization": full_standardization,
        "lambda": {
            "lambda_max": lambda_max,
            "rule": "max_abs_interaction_gradient_after_local_unpenalized_start",
        },
        "cv": {
            "selected_index": selected_index,
            "selected_lambda": float(lambda_values[selected_index]),
            "lambda_path": lambda_values.tolist(),
            "fold_validation_losses": fold_validation_losses.tolist(),
            "aggregated_validation_losses": aggregated_losses.tolist(),
            "fold_reports": fold_reports,
        },
        "optimizer": {
            "unpenalized_start": unpenalized_report,
            "lambda_fits": final_reports,
            "selected_lambda_fit": final_reports[selected_index],
        },
    }
