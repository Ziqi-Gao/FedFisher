"""dsLassoCov-style federated logistic LASSO for effect-modifier recovery.

This is a Python benchmark implementation, not the official R/DataSHIELD
dsLassoCov software.  Individual training rows remain in ``FederatedSite``
objects.  Optimization aggregates only site-level loss and gradient vectors.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn


METHOD_NAME = "dsLassoCov-style federated logistic LASSO"
IMPLEMENTATION_LABEL = "Python benchmark; not official R/DataSHIELD dsLassoCov"

# Observational only: unchanged callers have no communication observer, and
# objective/optimizer values and return contracts are unaffected.
_COMMUNICATION_OBSERVER = ContextVar("dslassocov_communication_observer", default=None)


@contextmanager
def observe_communication(observer):
    """Observe every numeric aggregation payload during one scoped fit.

    Events describe a single-process simulation, not real network transfers.
    Context-local state is restored on success or failure, including nesting.
    """
    if observer is not None and not callable(observer):
        raise ValueError("communication observer must be callable or None")
    token = _COMMUNICATION_OBSERVER.set(observer)
    try:
        yield
    finally:
        _COMMUNICATION_OBSERVER.reset(token)


def _payload_hash_update(digest, site_idx, count, *arrays):
    digest.update(np.asarray([site_idx, count], dtype="<i8").tobytes())
    for array in arrays:
        digest.update(np.asarray(array, dtype="<f8", order="C").tobytes())

VARIANT_CONTRACT = {
    "control": {
        "site_gaussian_shift": False,
        "intervention_scope": "global",
    },
    "site_gaussian_shift": {
        "site_gaussian_shift": True,
        "intervention_scope": "global",
    },
    "local_permutation": {
        "site_gaussian_shift": False,
        "intervention_scope": "site",
    },
    "site_gaussian_local_permutation": {
        "site_gaussian_shift": True,
        "intervention_scope": "site",
    },
}


@dataclass(frozen=True)
class RawSite:
    """One site's original-scale design without an intercept column."""

    features: np.ndarray
    labels: np.ndarray
    site_idx: int


@dataclass(frozen=True)
class FederatedSite:
    """One site's standardized design, including an intercept column."""

    design: np.ndarray
    labels: np.ndarray
    site_idx: int


@dataclass(frozen=True)
class OptimizerConfig:
    max_iter: int = 500
    tolerance: float = 1e-6
    initial_step: float = 1.0
    backtrack_factor: float = 0.5
    min_step: float = 1e-12


class FederatedLogisticLassoModel(nn.Module):
    """Two-logit PyTorch wrapper for the fitted original-scale linear score."""

    def __init__(self, intercept: float, coefficients: Sequence[float]):
        super().__init__()
        coefficient_tensor = torch.as_tensor(coefficients, dtype=torch.float32)
        self.coefficients = nn.Parameter(coefficient_tensor, requires_grad=False)
        self.intercept = nn.Parameter(
            torch.tensor(float(intercept), dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, inputs):
        score = inputs.matmul(self.coefficients) + self.intercept
        return torch.stack((torch.zeros_like(score), score), dim=1)


def get_variant_contract(variant: str) -> Dict[str, object]:
    try:
        return dict(VARIANT_CONTRACT[variant])
    except KeyError as error:
        raise ValueError(
            "variant must be one of: %s" % ", ".join(VARIANT_CONTRACT)
        ) from error


def tensor_datasets_to_raw_sites(
    site_datasets: Sequence[torch.utils.data.TensorDataset],
    covariate_dim: int,
) -> List[RawSite]:
    expected_dim = 1 + 2 * covariate_dim
    raw_sites = []
    for site_idx, dataset in enumerate(site_datasets):
        if not hasattr(dataset, "tensors") or len(dataset.tensors) != 2:
            raise TypeError("training sites must be two-tensor TensorDataset objects")
        features = dataset.tensors[0].detach().cpu().numpy()
        labels = dataset.tensors[1].detach().cpu().numpy()
        if features.ndim != 2 or features.shape[1] != expected_dim:
            raise ValueError(
                "site %d should have %d features, got %s"
                % (site_idx, expected_dim, features.shape)
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError("site %d has incompatible features and labels" % site_idx)
        if features.shape[0] == 0:
            raise ValueError("every training site must contain at least one row")
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError("site %d labels must be binary 0/1" % site_idx)
        raw_sites.append(
            RawSite(
                features=np.asarray(features, dtype=np.float64, order="C"),
                labels=np.asarray(labels, dtype=np.float64, order="C"),
                site_idx=site_idx,
            )
        )
    if not raw_sites:
        raise ValueError("at least one training site is required")
    return raw_sites


def training_fingerprint(raw_sites: Sequence[RawSite]) -> str:
    digest = hashlib.sha256()
    digest.update(b"fedfisher-dslassocov-training-v1\0")
    for site in raw_sites:
        digest.update(str(site.site_idx).encode("ascii") + b"\0")
        digest.update(np.asarray(site.features.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(site.features, dtype="<f8", order="C").tobytes())
        digest.update(np.asarray(site.labels, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()


def dataset_fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(b"fedfisher-dslassocov-dataset-v1\0")
    if hasattr(dataset, "tensors"):
        tensors = dataset.tensors
        for tensor in tensors:
            array = tensor.detach().cpu().numpy()
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(np.ascontiguousarray(array).tobytes())
    else:
        for index in range(len(dataset)):
            features, label = dataset[index]
            digest.update(np.ascontiguousarray(features.detach().cpu().numpy()).tobytes())
            digest.update(np.asarray([int(label)], dtype="<i8").tobytes())
    return digest.hexdigest()


def _validate_raw_sites(raw_sites: Sequence[RawSite]) -> int:
    if not raw_sites:
        raise ValueError("at least one site is required")
    feature_dim = raw_sites[0].features.shape[1]
    for site in raw_sites:
        if site.features.ndim != 2 or site.features.shape[1] != feature_dim:
            raise ValueError("all sites must have the same feature dimension")
        if site.labels.shape != (site.features.shape[0],):
            raise ValueError("site labels must match its feature rows")
        if not np.isfinite(site.features).all() or not np.isfinite(site.labels).all():
            raise ValueError("site features and labels must be finite")
    return feature_dim


def aggregate_standardization(
    raw_sites: Sequence[RawSite],
    scale_floor: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Compute global moments from site-level sums and sums of squares."""
    feature_dim = _validate_raw_sites(raw_sites)
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    observer = _COMMUNICATION_OBSERVER.get()
    payload_hash = hashlib.sha256(b"dslassocov-standardization-payload-v1\0") if observer is not None else None
    total_count = 0
    feature_sum = np.zeros(feature_dim, dtype=np.float64)
    feature_square_sum = np.zeros(feature_dim, dtype=np.float64)
    site_counts = []
    for site in raw_sites:
        site_counts.append(int(site.features.shape[0]))
        total_count += site.features.shape[0]
        site_sum = site.features.sum(axis=0, dtype=np.float64)
        site_square_sum = np.square(site.features).sum(axis=0, dtype=np.float64)
        feature_sum += site_sum
        feature_square_sum += site_square_sum
        if observer is not None:
            _payload_hash_update(payload_hash, site.site_idx, site.features.shape[0], site_sum, site_square_sum)
    means = feature_sum / total_count
    variances = np.maximum(feature_square_sum / total_count - np.square(means), 0.0)
    raw_scales = np.sqrt(variances)
    constant_mask = raw_scales < scale_floor
    scales = raw_scales.copy()
    scales[constant_mask] = 1.0
    report = {
        "aggregation": "site_count_sum_sumsq_only",
        "num_examples": int(total_count),
        "site_counts": site_counts,
        "scale_floor": float(scale_floor),
        "constant_feature_indices": np.flatnonzero(constant_mask).astype(int).tolist(),
    }
    if observer is not None:
        for site in raw_sites:
            _payload_hash_update(payload_hash, site.site_idx, site.features.shape[0], means, scales)
        observer({
            "kind": "standardization",
            "num_sites": len(raw_sites),
            "vector_elements": int(feature_dim),
            "upload_bytes": len(raw_sites) * (16 + 16 * feature_dim),
            "download_bytes": len(raw_sites) * 16 * feature_dim,
            "payload_sha256": payload_hash.hexdigest(),
        })
    return means, scales, report


def standardize_sites(
    raw_sites: Sequence[RawSite],
    means: np.ndarray,
    scales: np.ndarray,
) -> List[FederatedSite]:
    feature_dim = _validate_raw_sites(raw_sites)
    if means.shape != (feature_dim,) or scales.shape != (feature_dim,):
        raise ValueError("standardization vectors have the wrong dimension")
    if np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise ValueError("standardization scales must be finite and positive")
    standardized_sites = []
    for site in raw_sites:
        design = np.empty(
            (site.features.shape[0], feature_dim + 1),
            dtype=np.float64,
        )
        design[:, 0] = 1.0
        design[:, 1:] = (site.features - means) / scales
        standardized_sites.append(
            FederatedSite(design=design, labels=site.labels, site_idx=site.site_idx)
        )
    return standardized_sites


def backtransform_coefficients(
    standardized_coefficients: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> Tuple[float, np.ndarray]:
    if standardized_coefficients.shape != (means.size + 1,):
        raise ValueError("coefficient and standardization dimensions do not match")
    coefficients = standardized_coefficients[1:] / scales
    intercept = standardized_coefficients[0] - float(np.dot(coefficients, means))
    return float(intercept), coefficients


def _aggregate_loss_and_gradient(
    sites: Sequence[FederatedSite],
    coefficients: np.ndarray,
    include_gradient: bool,
):
    total_count = sum(site.design.shape[0] for site in sites)
    if total_count <= 0:
        raise ValueError("federated objective requires at least one row")
    observer = _COMMUNICATION_OBSERVER.get()
    payload_hash = hashlib.sha256(b"dslassocov-objective-payload-v1\0") if observer is not None else None
    loss_sum = 0.0
    gradient_sum = np.zeros_like(coefficients) if include_gradient else None
    site_summaries = []
    for site in sites:
        if observer is not None:
            _payload_hash_update(payload_hash, site.site_idx, site.design.shape[0], coefficients)
        scores = site.design.dot(coefficients)
        site_loss_sum = float(
            np.sum(np.logaddexp(0.0, scores) - site.labels * scores)
        )
        loss_sum += site_loss_sum
        if include_gradient:
            probabilities = np.empty_like(scores)
            positive = scores >= 0.0
            probabilities[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
            negative_exp = np.exp(scores[~positive])
            probabilities[~positive] = negative_exp / (1.0 + negative_exp)
            site_gradient_sum = site.design.T.dot(probabilities - site.labels)
            gradient_sum += site_gradient_sum
            site_summaries.append(
                {
                    "site_idx": int(site.site_idx),
                    "num_examples": int(site.design.shape[0]),
                    "loss_sum": site_loss_sum,
                }
            )
        if observer is not None:
            values = (site_loss_sum, site_gradient_sum) if include_gradient else (site_loss_sum,)
            _payload_hash_update(payload_hash, site.site_idx, site.design.shape[0], *values)
    loss = loss_sum / total_count
    if observer is not None:
        observer({
            "kind": "loss_gradient" if include_gradient else "loss_only",
            "num_sites": len(sites),
            "vector_elements": int(coefficients.size),
            "upload_bytes": len(sites) * (24 + (8 * coefficients.size if include_gradient else 0)),
            "download_bytes": len(sites) * 8 * coefficients.size,
            "payload_sha256": payload_hash.hexdigest(),
        })
    if include_gradient:
        return loss, gradient_sum / total_count, site_summaries
    return loss


def aggregate_loss_and_gradient(
    sites: Sequence[FederatedSite], coefficients: np.ndarray
):
    """Public audit helper returning the exact site-aggregated smooth objective."""
    return _aggregate_loss_and_gradient(sites, coefficients, include_gradient=True)


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _penalized_objective(
    sites: Sequence[FederatedSite],
    coefficients: np.ndarray,
    penalty_start: int,
    lambda_value: float,
) -> float:
    smooth_loss = _aggregate_loss_and_gradient(
        sites, coefficients, include_gradient=False
    )
    return float(
        smooth_loss + lambda_value * np.abs(coefficients[penalty_start:]).sum()
    )


def proximal_gradient_fit(
    sites: Sequence[FederatedSite],
    penalty_start: int,
    lambda_value: float,
    config: OptimizerConfig,
    initial_coefficients: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """FISTA with adaptive restart and federated site-level backtracking."""
    parameter_dim = sites[0].design.shape[1]
    if not 0 <= penalty_start <= parameter_dim:
        raise ValueError("penalty_start is outside the coefficient vector")
    if lambda_value < 0.0:
        raise ValueError("lambda_value must be nonnegative")
    if config.max_iter < 1 or config.tolerance <= 0.0:
        raise ValueError("optimizer max_iter and tolerance must be positive")
    if not 0.0 < config.backtrack_factor < 1.0:
        raise ValueError("backtrack_factor must lie in (0, 1)")
    if config.initial_step <= 0.0 or config.min_step <= 0.0:
        raise ValueError("optimizer step sizes must be positive")

    if initial_coefficients is None:
        coefficients = np.zeros(parameter_dim, dtype=np.float64)
    else:
        coefficients = np.asarray(initial_coefficients, dtype=np.float64).copy()
        if coefficients.shape != (parameter_dim,):
            raise ValueError("initial coefficient vector has the wrong dimension")
    if not np.isfinite(coefficients).all():
        raise ValueError("initial coefficient vector must be finite")

    extrapolated = coefficients.copy()
    momentum = 1.0
    step_size = float(config.initial_step)
    previous_objective = _penalized_objective(
        sites, coefficients, penalty_start, lambda_value
    )
    initial_objective = previous_objective
    total_backtracks = 0
    gradient_rounds = 0
    loss_rounds = 1
    converged = False
    max_update = float("inf")
    restart_count = 0

    for iteration in range(1, config.max_iter + 1):
        restarted_this_iteration = False
        while True:
            smooth_loss, gradient, _ = _aggregate_loss_and_gradient(
                sites, extrapolated, include_gradient=True
            )
            gradient_rounds += 1
            trial_step = step_size
            while True:
                candidate = extrapolated - trial_step * gradient
                candidate[penalty_start:] = _soft_threshold(
                    candidate[penalty_start:], trial_step * lambda_value
                )
                candidate_smooth_loss = _aggregate_loss_and_gradient(
                    sites, candidate, include_gradient=False
                )
                loss_rounds += 1
                delta_from_extrapolated = candidate - extrapolated
                quadratic_bound = (
                    smooth_loss
                    + float(np.dot(gradient, delta_from_extrapolated))
                    + float(np.dot(delta_from_extrapolated, delta_from_extrapolated))
                    / (2.0 * trial_step)
                )
                if candidate_smooth_loss <= quadratic_bound + 1e-12:
                    break
                trial_step *= config.backtrack_factor
                total_backtracks += 1
                if trial_step < config.min_step:
                    raise RuntimeError("proximal-gradient backtracking reached min_step")

            candidate_objective = float(
                candidate_smooth_loss
                + lambda_value * np.abs(candidate[penalty_start:]).sum()
            )
            if (
                candidate_objective > previous_objective + 1e-12
                and momentum > 1.0
                and not restarted_this_iteration
            ):
                extrapolated = coefficients.copy()
                momentum = 1.0
                restarted_this_iteration = True
                restart_count += 1
                continue
            break

        step_size = trial_step
        update = candidate - coefficients
        max_update = float(np.max(np.abs(update)))
        threshold = config.tolerance * (
            1.0 + float(np.max(np.abs(coefficients)))
        )
        previous_coefficients = coefficients
        coefficients = candidate
        previous_objective = candidate_objective
        if max_update <= threshold:
            _, checkpoint_gradient, _ = _aggregate_loss_and_gradient(
                sites, coefficients, include_gradient=True
            )
            gradient_rounds += 1
            checkpoint_proximal = coefficients - step_size * checkpoint_gradient
            checkpoint_proximal[penalty_start:] = _soft_threshold(
                checkpoint_proximal[penalty_start:], step_size * lambda_value
            )
            checkpoint_mapping_inf = float(
                np.max(
                    np.abs(
                        (coefficients - checkpoint_proximal) / step_size
                    )
                )
            )
            checkpoint_unpenalized_kkt = (
                float(np.max(np.abs(checkpoint_gradient[:penalty_start])))
                if penalty_start
                else 0.0
            )
            checkpoint_penalized_coefficients = coefficients[penalty_start:]
            checkpoint_penalized_gradient = checkpoint_gradient[penalty_start:]
            checkpoint_nonzero = checkpoint_penalized_coefficients != 0.0
            checkpoint_penalized_residual = np.maximum(
                np.abs(checkpoint_penalized_gradient) - lambda_value,
                0.0,
            )
            checkpoint_penalized_residual[checkpoint_nonzero] = np.abs(
                checkpoint_penalized_gradient[checkpoint_nonzero]
                + lambda_value
                * np.sign(checkpoint_penalized_coefficients[checkpoint_nonzero])
            )
            checkpoint_penalized_kkt = (
                float(np.max(checkpoint_penalized_residual))
                if checkpoint_penalized_residual.size
                else 0.0
            )
            checkpoint_kkt_inf = max(
                checkpoint_unpenalized_kkt, checkpoint_penalized_kkt
            )
            checkpoint_threshold = config.tolerance * (
                1.0 + float(np.max(np.abs(coefficients)))
            )
            if (
                checkpoint_mapping_inf <= checkpoint_threshold
                and checkpoint_kkt_inf <= checkpoint_threshold
            ):
                converged = True
                break
            extrapolated = coefficients.copy()
            momentum = 1.0
            restart_count += 1
            continue

        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = coefficients + (
            (momentum - 1.0) / next_momentum
        ) * (coefficients - previous_coefficients)
        momentum = next_momentum

    _, final_gradient, _ = _aggregate_loss_and_gradient(
        sites, coefficients, include_gradient=True
    )
    gradient_rounds += 1
    proximal_point = coefficients - step_size * final_gradient
    proximal_point[penalty_start:] = _soft_threshold(
        proximal_point[penalty_start:], step_size * lambda_value
    )
    proximal_gradient_mapping_inf = float(
        np.max(np.abs((coefficients - proximal_point) / step_size))
    )
    unpenalized_kkt_inf = (
        float(np.max(np.abs(final_gradient[:penalty_start])))
        if penalty_start
        else 0.0
    )
    penalized_coefficients = coefficients[penalty_start:]
    penalized_gradient = final_gradient[penalty_start:]
    nonzero = penalized_coefficients != 0.0
    penalized_kkt_residual = np.maximum(
        np.abs(penalized_gradient) - lambda_value,
        0.0,
    )
    penalized_kkt_residual[nonzero] = np.abs(
        penalized_gradient[nonzero]
        + lambda_value * np.sign(penalized_coefficients[nonzero])
    )
    penalized_kkt_inf = (
        float(np.max(penalized_kkt_residual))
        if penalized_kkt_residual.size
        else 0.0
    )
    kkt_residual_inf = max(unpenalized_kkt_inf, penalized_kkt_inf)
    stationarity_threshold = config.tolerance * (
        1.0 + float(np.max(np.abs(coefficients)))
    )
    converged = bool(
        converged
        and proximal_gradient_mapping_inf <= stationarity_threshold
        and kkt_residual_inf <= stationarity_threshold
    )

    report = {
        "iterations": int(iteration),
        "converged": bool(converged),
        "initial_objective": float(initial_objective),
        "final_objective": float(previous_objective),
        "max_coefficient_update": float(max_update),
        "proximal_gradient_mapping_inf": proximal_gradient_mapping_inf,
        "kkt_residual_inf": float(kkt_residual_inf),
        "unpenalized_kkt_residual_inf": float(unpenalized_kkt_inf),
        "penalized_kkt_residual_inf": float(penalized_kkt_inf),
        "stationarity_threshold": float(stationarity_threshold),
        "final_step_size": float(step_size),
        "backtracking_loss_rounds": int(loss_rounds),
        "gradient_rounds": int(gradient_rounds),
        "total_backtracks": int(total_backtracks),
        "adaptive_restarts": int(restart_count),
        "site_level_messages_per_gradient_round": int(len(sites)),
    }
    return coefficients, report


def deterministic_site_folds(
    raw_sites: Sequence[RawSite],
    num_folds: int,
    cv_seed: int,
) -> List[np.ndarray]:
    """Create deterministic, outcome-stratified folds independently per site."""
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    fold_assignments = []
    for site in raw_sites:
        if site.features.shape[0] < num_folds:
            raise ValueError("each site needs at least num_folds rows")
        assignments = np.full(site.features.shape[0], -1, dtype=np.int64)
        for label in (0, 1):
            indices = np.flatnonzero(site.labels == label)
            rng = np.random.default_rng(
                int(cv_seed) + (site.site_idx + 1) * 104729 + label * 1009
            )
            rng.shuffle(indices)
            for position, row_idx in enumerate(indices):
                assignments[row_idx] = position % num_folds
        if np.any(assignments < 0):
            raise RuntimeError("site-local fold assignment left rows unassigned")
        fold_assignments.append(assignments)
    return fold_assignments


def _subset_raw_sites(
    raw_sites: Sequence[RawSite],
    fold_assignments: Sequence[np.ndarray],
    fold_idx: int,
    validation: bool,
) -> List[RawSite]:
    subsets = []
    for site, assignments in zip(raw_sites, fold_assignments):
        mask = assignments == fold_idx
        if not validation:
            mask = ~mask
        if not np.any(mask):
            raise ValueError("every site must contribute rows to each CV partition")
        subsets.append(
            RawSite(
                features=site.features[mask],
                labels=site.labels[mask],
                site_idx=site.site_idx,
            )
        )
    return subsets


def _unpenalized_start(
    sites: Sequence[FederatedSite],
    penalty_start: int,
    optimizer_config: OptimizerConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    unpenalized_sites = [
        FederatedSite(
            design=site.design[:, :penalty_start],
            labels=site.labels,
            site_idx=site.site_idx,
        )
        for site in sites
    ]
    unpenalized_coefficients, report = proximal_gradient_fit(
        unpenalized_sites,
        penalty_start=penalty_start,
        lambda_value=0.0,
        config=optimizer_config,
    )
    if not report["converged"]:
        raise RuntimeError(
            "unpenalized covariate-adjustment fit failed its stationarity check"
        )
    coefficients = np.zeros(sites[0].design.shape[1], dtype=np.float64)
    coefficients[:penalty_start] = unpenalized_coefficients
    return coefficients, report


def _lambda_path(
    standardized_sites: Sequence[FederatedSite],
    penalty_start: int,
    lambda_count: int,
    lambda_min_ratio: float,
    optimizer_config: OptimizerConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if lambda_count < 2:
        raise ValueError("lambda_count must be at least 2")
    if not 0.0 < lambda_min_ratio < 1.0:
        raise ValueError("lambda_min_ratio must lie in (0, 1)")
    starting_coefficients, start_report = _unpenalized_start(
        standardized_sites, penalty_start, optimizer_config
    )
    _, gradient, _ = aggregate_loss_and_gradient(
        standardized_sites, starting_coefficients
    )
    lambda_max = float(np.max(np.abs(gradient[penalty_start:])))
    if not np.isfinite(lambda_max) or lambda_max <= 0.0:
        raise RuntimeError("could not derive a positive finite lambda_max")
    path = np.geomspace(
        lambda_max,
        lambda_max * lambda_min_ratio,
        num=lambda_count,
        dtype=np.float64,
    )
    return path, starting_coefficients, {
        "lambda_max_rule": (
            "max absolute penalized logistic gradient after optimizing the "
            "unpenalized block at zero interactions"
        ),
        "lambda_max": lambda_max,
        "unpenalized_start": start_report,
    }


def _fit_path(
    sites: Sequence[FederatedSite],
    penalty_start: int,
    lambda_values: Sequence[float],
    initial_coefficients: np.ndarray,
    optimizer_config: OptimizerConfig,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    coefficients = initial_coefficients.copy()
    path_coefficients = []
    reports = []
    for lambda_value in lambda_values:
        coefficients, report = proximal_gradient_fit(
            sites,
            penalty_start=penalty_start,
            lambda_value=float(lambda_value),
            config=optimizer_config,
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


def _fit_fingerprint(
    intercept: float,
    coefficients: np.ndarray,
    selected_lambda: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"fedfisher-dslassocov-fit-v1\0")
    digest.update(np.asarray([intercept, selected_lambda], dtype="<f8").tobytes())
    digest.update(np.asarray(coefficients, dtype="<f8").tobytes())
    return digest.hexdigest()


def federated_logistic_lasso_cv(
    raw_sites: Sequence[RawSite],
    covariate_dim: int,
    num_folds: int = 5,
    lambda_count: int = 30,
    lambda_min_ratio: float = 1e-3,
    cv_seed: int = 4000037,
    scale_floor: float = 1e-12,
    optimizer_config: OptimizerConfig = OptimizerConfig(),
) -> Dict[str, object]:
    """Fit training-only CV and a final federated logistic LASSO path."""
    feature_dim = _validate_raw_sites(raw_sites)
    expected_feature_dim = 1 + 2 * covariate_dim
    if feature_dim != expected_feature_dim:
        raise ValueError(
            "expected [A, raw X, A*X] dimension %d, got %d"
            % (expected_feature_dim, feature_dim)
        )
    penalty_start = 2 + covariate_dim
    fold_assignments = deterministic_site_folds(raw_sites, num_folds, cv_seed)

    full_means, full_scales, full_standardization_report = aggregate_standardization(
        raw_sites, scale_floor=scale_floor
    )
    full_standardized_sites = standardize_sites(raw_sites, full_means, full_scales)
    lambda_values, full_start, lambda_report = _lambda_path(
        full_standardized_sites,
        penalty_start,
        lambda_count,
        lambda_min_ratio,
        optimizer_config,
    )

    fold_validation_losses = np.zeros((num_folds, lambda_count), dtype=np.float64)
    fold_counts = []
    fold_reports = []
    for fold_idx in range(num_folds):
        train_sites = _subset_raw_sites(
            raw_sites, fold_assignments, fold_idx, validation=False
        )
        validation_sites = _subset_raw_sites(
            raw_sites, fold_assignments, fold_idx, validation=True
        )
        fold_means, fold_scales, fold_standardization_report = (
            aggregate_standardization(train_sites, scale_floor=scale_floor)
        )
        standardized_train_sites = standardize_sites(
            train_sites, fold_means, fold_scales
        )
        standardized_validation_sites = standardize_sites(
            validation_sites, fold_means, fold_scales
        )
        fold_start, unpenalized_report = _unpenalized_start(
            standardized_train_sites, penalty_start, optimizer_config
        )
        path_coefficients, optimizer_reports = _fit_path(
            standardized_train_sites,
            penalty_start,
            lambda_values,
            fold_start,
            optimizer_config,
        )
        for lambda_idx, coefficients in enumerate(path_coefficients):
            fold_validation_losses[fold_idx, lambda_idx] = (
                _aggregate_loss_and_gradient(
                    standardized_validation_sites,
                    coefficients,
                    include_gradient=False,
                )
            )
        validation_count = sum(site.features.shape[0] for site in validation_sites)
        fold_counts.append(validation_count)
        fold_reports.append(
            {
                "fold_idx": fold_idx,
                "training_examples": int(
                    sum(site.features.shape[0] for site in train_sites)
                ),
                "validation_examples": int(validation_count),
                "standardization": fold_standardization_report,
                "unpenalized_start": unpenalized_report,
                "lambda_fits": optimizer_reports,
            }
        )

    fold_weights = np.asarray(fold_counts, dtype=np.float64)
    aggregated_validation_losses = np.average(
        fold_validation_losses, axis=0, weights=fold_weights
    )
    minimum_loss = float(np.min(aggregated_validation_losses))
    tied_indices = np.flatnonzero(
        np.isclose(
            aggregated_validation_losses,
            minimum_loss,
            rtol=0.0,
            atol=1e-12,
        )
    )
    selected_index = int(tied_indices[0])
    selected_lambda = float(lambda_values[selected_index])

    final_path_coefficients, final_reports = _fit_path(
        full_standardized_sites,
        penalty_start,
        lambda_values,
        full_start,
        optimizer_config,
    )
    standardized_coefficients = final_path_coefficients[selected_index]
    intercept, original_coefficients = backtransform_coefficients(
        standardized_coefficients, full_means, full_scales
    )

    return {
        "method": METHOD_NAME,
        "implementation": IMPLEMENTATION_LABEL,
        "objective": (
            "sample-weighted federated Bernoulli logistic loss plus L1 on "
            "A-by-X interaction coefficients only"
        ),
        "aggregation_contract": (
            "individual rows stay at sites; coordinator sums site loss and "
            "gradient contributions weighted by sample count"
        ),
        "unpenalized_columns": ["intercept", "A", "raw_X1..raw_X%d" % covariate_dim],
        "penalized_columns": "A_by_X1..A_by_X%d" % covariate_dim,
        "penalty_start": int(penalty_start),
        "num_penalized": int(covariate_dim),
        "cv": {
            "uses_test_data": False,
            "num_folds": int(num_folds),
            "fold_assignment": (
                "deterministic outcome-stratified folds constructed independently within each site"
            ),
            "cv_seed": int(cv_seed),
            "lambda_count": int(lambda_count),
            "lambda_min_ratio": float(lambda_min_ratio),
            "lambda_path": lambda_values.tolist(),
            "lambda_path_report": lambda_report,
            "selection_rule": (
                "minimum sample-count-weighted validation logistic loss; "
                "ties within 1e-12 choose the largest lambda"
            ),
            "fold_validation_losses": fold_validation_losses.tolist(),
            "fold_validation_counts": [int(value) for value in fold_counts],
            "aggregated_validation_losses": aggregated_validation_losses.tolist(),
            "selected_index": selected_index,
            "selected_lambda": selected_lambda,
            "fold_reports": fold_reports,
        },
        "optimizer": {
            "algorithm": "FISTA proximal gradient with adaptive restart and backtracking",
            "max_iter": int(optimizer_config.max_iter),
            "tolerance": float(optimizer_config.tolerance),
            "initial_step": float(optimizer_config.initial_step),
            "backtrack_factor": float(optimizer_config.backtrack_factor),
            "min_step": float(optimizer_config.min_step),
            "final_lambda_fits": final_reports,
            "selected_lambda_fit": final_reports[selected_index],
            "all_final_path_fits_converged": bool(
                all(report["converged"] for report in final_reports)
            ),
        },
        "standardization": {
            "fit_scope": "training rows only via site-level count/sum/sumsq",
            "feature_order": ["A", "raw_X1..raw_X%d" % covariate_dim, "A_by_X1..A_by_X%d" % covariate_dim],
            "means": full_means.tolist(),
            "scales": full_scales.tolist(),
            "report": full_standardization_report,
            "backtransform": (
                "beta_original=beta_standardized/scale; "
                "intercept_original=intercept_standardized-dot(beta_original,mean)"
            ),
        },
        "standardized_coefficients": standardized_coefficients.tolist(),
        "original_intercept": intercept,
        "original_coefficients": original_coefficients.tolist(),
        "fit_fingerprint": _fit_fingerprint(
            intercept, original_coefficients, selected_lambda
        ),
    }


def interaction_coefficient_recovery(
    fit_result: Dict[str, object],
    covariate_dim: int,
    signal_dim: int,
) -> Dict[str, object]:
    """Clearly auxiliary recovery based on absolute interaction coefficients."""
    from utils.prediction_intervention import evaluate_signal_recovery

    coefficients = np.asarray(fit_result["original_coefficients"], dtype=np.float64)
    interaction_coefficients = coefficients[1 + covariate_dim :]
    if interaction_coefficients.shape != (covariate_dim,):
        raise ValueError("fitted interaction block has the wrong dimension")
    recovery = evaluate_signal_recovery(
        np.abs(interaction_coefficients),
        signal_dim=signal_dim,
        metric="absolute_interaction_coefficient_auxiliary",
        signal_indices=range(signal_dim),
        candidate_indices=range(covariate_dim),
    )
    return {
        "diagnostic_role": "auxiliary_not_primary",
        "metric": "absolute_interaction_coefficient_recovery",
        "absolute_coefficients": np.abs(interaction_coefficients).tolist(),
        "signed_coefficients": interaction_coefficients.tolist(),
        "recovery": recovery,
    }


def json_safe(value):
    """Convert NumPy scalar values recursively for durable JSON output."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
