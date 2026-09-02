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

import json
from typing import Dict, List, Optional, Tuple

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
MOMENTUM_RESET_POLISH_RULE = (
    "reset_fista_momentum_when_legacy_flag_false_but_final_prox_and_kkt_pass"
)
MOMENTUM_RESET_POLISH_SCOPE = "unpenalized_starts_and_lambda_path_fits"
DEFAULT_MOMENTUM_RESET_POLISH_MAX_ITER = 50
STATIONARITY_FAILURE_DIAGNOSTIC_SCHEMA_VERSION = 1


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
    polish_max_iter: int = DEFAULT_MOMENTUM_RESET_POLISH_MAX_ITER,
    *,
    stage: str = "unpenalized_start",
    fold_idx: Optional[int] = None,
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
    reduced_coefficients, report = _momentum_reset_polish(
        reduced_sites,
        penalty_start,
        0.0,
        reduced_coefficients,
        report,
        optimizer,
        polish_max_iter,
    )
    if not report["converged"]:
        raise StationarityFailure(
            report,
            stage=stage,
            site_idx=int(standardized_sites[0].site_idx),
            fold_idx=fold_idx,
            lambda_value=0.0,
        )
    coefficients = np.zeros(standardized_sites[0].design.shape[1], dtype=np.float64)
    coefficients[:penalty_start] = reduced_coefficients
    return coefficients, report


def _passes_final_stationarity(report: Dict[str, object]) -> bool:
    threshold = float(report["stationarity_threshold"])
    proximal = float(report["proximal_gradient_mapping_inf"])
    kkt = float(report["kkt_residual_inf"])
    return bool(
        np.isfinite([threshold, proximal, kkt]).all()
        and threshold > 0.0
        and proximal <= threshold
        and kkt <= threshold
    )


def _diagnostic_value(value):
    """Return a deterministic JSON-safe diagnostic value."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            str(key): _diagnostic_value(child)
            for key, child in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(child) for child in value]
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stationarity_failure_diagnostic(
    report: Dict[str, object],
    *,
    stage: str,
    site_idx: int,
    fold_idx: Optional[int] = None,
    lambda_index: Optional[int] = None,
    lambda_value: Optional[float] = None,
) -> Dict[str, object]:
    """Describe a rejected fit without changing the convergence boundary."""
    diagnostic = {
        "schema_version": STATIONARITY_FAILURE_DIAGNOSTIC_SCHEMA_VERSION,
        "error_type": "one_shot_logistic_stationarity_failure",
        "run_id": None,
        "family": None,
        "variant": None,
        "seed": None,
        "split": None,
        "model": None,
        "cell": None,
        "scheduler_job_id": None,
        "scheduler_attempt": None,
        "stage": str(stage),
        "site_idx": int(site_idx),
        "fold_idx": None if fold_idx is None else int(fold_idx),
        "lambda_index": None if lambda_index is None else int(lambda_index),
        "lambda": None if lambda_value is None else float(lambda_value),
    }
    for key in (
        "iterations",
        "converged",
        "initial_objective",
        "final_objective",
        "max_coefficient_update",
        "proximal_gradient_mapping_inf",
        "kkt_residual_inf",
        "unpenalized_kkt_residual_inf",
        "penalized_kkt_residual_inf",
        "stationarity_threshold",
        "final_step_size",
        "total_backtracks",
        "adaptive_restarts",
    ):
        diagnostic[key] = report.get(key)
    diagnostic["momentum_reset_polish"] = report.get(
        "momentum_reset_polish"
    )
    return _diagnostic_value(diagnostic)


def _stationarity_failure_message(diagnostic: Dict[str, object]) -> str:
    return "one-shot logistic stationarity failure: " + json.dumps(
        diagnostic,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class StationarityFailure(RuntimeError):
    """A fail-closed stationarity rejection with canonical JSON context."""

    _CONTEXT_FIELDS = frozenset(
        (
            "run_id",
            "family",
            "variant",
            "seed",
            "split",
            "model",
            "cell",
            "scheduler_job_id",
            "scheduler_attempt",
        )
    )

    def __init__(
        self,
        report: Dict[str, object],
        *,
        stage: str,
        site_idx: int,
        fold_idx: Optional[int] = None,
        lambda_index: Optional[int] = None,
        lambda_value: Optional[float] = None,
    ):
        self.diagnostic = stationarity_failure_diagnostic(
            report,
            stage=stage,
            site_idx=site_idx,
            fold_idx=fold_idx,
            lambda_index=lambda_index,
            lambda_value=lambda_value,
        )
        super().__init__(_stationarity_failure_message(self.diagnostic))

    def add_context(self, **context) -> None:
        unexpected = set(context).difference(self._CONTEXT_FIELDS)
        if unexpected:
            raise ValueError("unsupported stationarity diagnostic context")
        self.diagnostic.update(_diagnostic_value(context))
        self.args = (_stationarity_failure_message(self.diagnostic),)


def _momentum_reset_polish(
    standardized_sites,
    penalty_start: int,
    lambda_value: float,
    coefficients: np.ndarray,
    report: Dict[str, object],
    optimizer: OptimizerConfig,
    polish_max_iter: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Resolve a FISTA update-gate false negative without relaxing KKT."""
    report = dict(report)
    trigger_stationarity_passed = _passes_final_stationarity(report)
    if bool(report["converged"]) or not trigger_stationarity_passed:
        report["momentum_reset_polish"] = {
            "rule": MOMENTUM_RESET_POLISH_RULE,
            "attempted": False,
            "trigger_legacy_converged": bool(report["converged"]),
            "trigger_final_stationarity_passed": trigger_stationarity_passed,
            "max_iter": int(polish_max_iter),
        }
        return coefficients, report

    polish_optimizer = OptimizerConfig(
        max_iter=min(int(polish_max_iter), int(optimizer.max_iter)),
        tolerance=float(optimizer.tolerance),
        initial_step=float(optimizer.initial_step),
        backtrack_factor=float(optimizer.backtrack_factor),
        min_step=float(optimizer.min_step),
    )
    polished_coefficients, polish_report = proximal_gradient_fit(
        standardized_sites,
        penalty_start=penalty_start,
        lambda_value=float(lambda_value),
        config=polish_optimizer,
        initial_coefficients=coefficients,
    )
    polish_report = dict(polish_report)
    polish_stationarity_passed = _passes_final_stationarity(polish_report)
    coefficient_max_drift = float(
        np.max(np.abs(polished_coefficients - coefficients))
    )

    combined = dict(polish_report)
    combined["initial_objective"] = float(report["initial_objective"])
    combined["iterations"] = int(report["iterations"]) + int(
        polish_report["iterations"]
    )
    for key in (
        "backtracking_loss_rounds",
        "gradient_rounds",
        "total_backtracks",
        "adaptive_restarts",
    ):
        combined[key] = int(report[key]) + int(polish_report[key])
    combined["converged"] = bool(
        polish_report["converged"] and polish_stationarity_passed
    )
    combined["momentum_reset_polish"] = {
        "rule": MOMENTUM_RESET_POLISH_RULE,
        "attempted": True,
        "trigger_legacy_converged": bool(report["converged"]),
        "trigger_final_stationarity_passed": trigger_stationarity_passed,
        "max_iter": int(polish_optimizer.max_iter),
        "pre_iterations": int(report["iterations"]),
        "pre_proximal_gradient_mapping_inf": float(
            report["proximal_gradient_mapping_inf"]
        ),
        "pre_kkt_residual_inf": float(report["kkt_residual_inf"]),
        "pre_stationarity_threshold": float(report["stationarity_threshold"]),
        "pre_max_coefficient_update": float(report["max_coefficient_update"]),
        "post_iterations": int(polish_report["iterations"]),
        "post_converged": bool(polish_report["converged"]),
        "post_final_stationarity_passed": polish_stationarity_passed,
        "post_proximal_gradient_mapping_inf": float(
            polish_report["proximal_gradient_mapping_inf"]
        ),
        "post_kkt_residual_inf": float(polish_report["kkt_residual_inf"]),
        "post_stationarity_threshold": float(
            polish_report["stationarity_threshold"]
        ),
        "coefficient_max_drift": coefficient_max_drift,
    }
    return polished_coefficients, combined


def _fit_path(
    standardized_sites,
    penalty_start: int,
    lambda_values: np.ndarray,
    initial_coefficients: np.ndarray,
    optimizer: OptimizerConfig,
    polish_max_iter: int = DEFAULT_MOMENTUM_RESET_POLISH_MAX_ITER,
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
        coefficients, report = _momentum_reset_polish(
            standardized_sites,
            penalty_start,
            float(lambda_value),
            coefficients,
            report,
            optimizer,
            polish_max_iter,
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
    polish_max_iter: int = DEFAULT_MOMENTUM_RESET_POLISH_MAX_ITER,
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
    if polish_max_iter < 1 or polish_max_iter > optimizer.max_iter:
        raise ValueError("polish_max_iter must lie in [1, optimizer.max_iter]")

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
        full_standardized,
        penalty_start,
        optimizer,
        polish_max_iter,
        stage="full_unpenalized",
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
            standardized_training,
            penalty_start,
            optimizer,
            polish_max_iter,
            stage="fold_unpenalized",
            fold_idx=fold_idx,
        )
        coefficients, reports = _fit_path(
            standardized_training,
            penalty_start,
            lambda_values,
            start,
            optimizer,
            polish_max_iter,
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
        polish_max_iter,
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
            "momentum_reset_polish_rule": MOMENTUM_RESET_POLISH_RULE,
            "momentum_reset_polish_scope": MOMENTUM_RESET_POLISH_SCOPE,
            "momentum_reset_polish_max_iter": int(polish_max_iter),
            "stationarity_failure_diagnostic_schema_version": (
                STATIONARITY_FAILURE_DIAGNOSTIC_SCHEMA_VERSION
            ),
            "unpenalized_start": unpenalized_report,
            "lambda_fits": final_reports,
            "selected_lambda_fit": final_reports[selected_index],
        },
    }
