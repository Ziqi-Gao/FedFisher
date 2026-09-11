"""Homogeneous, summary-only adaptations of the DF R-learner CD objective.

Reference: Li et al., doi:10.1093/biomtc/ujag034 and the authors' implementation
https://github.com/Jin-hongLi/DF-R-learner/blob/main/DFRlearn_main_c1n100.R.
This is an independent Python adaptation, not a reproduction of their R code.

Each site estimates E[Y|X] and E[A|X] with strictly nested, local cross-fitting,
then regresses Y-m(X) on (A-e(X))*[1,X]. Only the local coefficient vector,
precision matrix and identifiers/count are accepted by the effect server.
The reference precision Z'Z/(RSS/n) already contains the sample-size factor.
It must NOT be multiplied by n again. The server imposes one common beta,
and optionally adds adaptive L1 regularization to its non-intercept entries.

Binary logistic nuisances, shared coefficients, and sparse-model EBIC are
explicit adaptations. The original DGP is unchanged: the linear probability
effect and homoskedastic precision are working models, not inference claims.
This module owns no datasets, filesystem paths, devices or launch surfaces.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math
import re
import time
from typing import Dict, Sequence, Tuple

import numpy as np

from effect_modifier_experiment.dslassocov_benchmark import (
    FederatedSite,
    OptimizerConfig,
    RawSite,
    proximal_gradient_fit,
)


@dataclass(frozen=True)
class RlearnerConfig:
    crossfit_folds: int = 5
    nuisance_cv_folds: int = 5
    nuisance_lambda_count: int = 30
    nuisance_lambda_max: float = 0.3
    nuisance_lambda_min_ratio: float = 0.001
    nuisance_max_iter: int = 4000
    nuisance_tolerance: float = 1e-6
    propensity_clip: float = 0.05
    constant_pseudocount: float = 0.5
    effect_lambda_count: int = 30
    effect_lambda_min_ratio: float = 0.0001
    adaptive_power: float = 0.5
    adaptive_epsilon: float = 1e-6
    ebic_gamma: float = 1.0
    effect_max_iter: int = 10000
    effect_tolerance: float = 1e-8
    condition_limit: float = 1e12
    residual_variance_floor: float = 1e-12


@dataclass(frozen=True)
class CausalSite:
    """Site-owned raw covariates, observed treatment and binary outcome."""

    covariates: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    site_idx: int


@dataclass(frozen=True)
class SiteSummary:
    """Complete upload accepted by the shared-effect server; no individual rows."""

    site_idx: int
    num_examples: int
    beta: np.ndarray
    precision: np.ndarray

    def payload_bytes(self) -> bytes:
        """Declared logical wire format: two int64 values, beta, dense precision."""
        return (
            np.asarray([self.site_idx, self.num_examples], dtype="<i8").tobytes()
            + np.asarray(self.beta, dtype="<f8").tobytes(order="C")
            + np.asarray(self.precision, dtype="<f8").tobytes(order="C")
        )


@dataclass(frozen=True)
class LinearProbabilityEffectModel:
    """Direct effect predictor; its output is not a classification logit.

    No clipping is performed: values outside [-1,1] reveal working-model
    extrapolation and must not be silently changed during evaluation.
    """

    coefficients: np.ndarray

    def __post_init__(self):
        values = np.array(self.coefficients, dtype=np.float64, copy=True)
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError("effect coefficients must be a finite intercept and slopes")
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)

    def predict_tau(self, raw_covariates: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_covariates, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] + 1 != self.coefficients.size
            or not np.isfinite(values).all()
        ):
            raise ValueError("effect prediction requires finite raw covariates [n,p]")
        return self.coefficients[0] + values.dot(self.coefficients[1:])


def _validate_config(config: RlearnerConfig) -> None:
    for name in (
        "crossfit_folds", "nuisance_cv_folds", "nuisance_lambda_count",
        "effect_lambda_count", "nuisance_max_iter", "effect_max_iter",
    ):
        value = getattr(config, name)
        minimum = 1 if name.endswith("max_iter") else 2
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("%s must be an integer >= %d" % (name, minimum))
    for name in (
        "nuisance_lambda_max", "nuisance_tolerance", "constant_pseudocount",
        "adaptive_power", "adaptive_epsilon", "effect_tolerance",
        "residual_variance_floor",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("%s must be positive and finite" % name)
    for name in ("nuisance_lambda_min_ratio", "effect_lambda_min_ratio"):
        if not 0.0 < getattr(config, name) < 1.0:
            raise ValueError("%s must lie strictly between zero and one" % name)
    if not 0 < config.propensity_clip < 0.5:
        raise ValueError("propensity_clip must lie in (0, 0.5)")
    if not np.isfinite(config.ebic_gamma) or not 0 <= config.ebic_gamma <= 1:
        raise ValueError("ebic_gamma must lie in [0,1]")
    if not np.isfinite(config.condition_limit) or config.condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")


def _validate_site(site: CausalSite) -> None:
    x = np.asarray(site.covariates)
    a = np.asarray(site.treatment)
    y = np.asarray(site.outcome)
    if (
        isinstance(site.site_idx, bool)
        or not isinstance(site.site_idx, (int, np.integer))
        or site.site_idx < 0
    ):
        raise ValueError("site_idx must be a nonnegative integer")
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("site covariates require a nonempty [n,p] matrix")
    if a.shape != (x.shape[0],) or y.shape != a.shape:
        raise ValueError("treatment/outcome must match site rows")
    if not all(np.isfinite(v).all() for v in (x, a, y)):
        raise ValueError("site data must be finite")
    if not np.isin(a, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("R-learner requires binary treatment and outcome")
    if np.unique(a).size != 2:
        raise ValueError("site has no observed treatment overlap")


def raw_sites_to_causal_sites(
    raw_sites: Sequence[RawSite], covariate_dim: int
) -> Sequence[CausalSite]:
    """Extract [A,X,A*X] without altering the existing generating mechanism."""
    if isinstance(covariate_dim, bool) or not isinstance(covariate_dim, int) or covariate_dim < 1:
        raise ValueError("covariate_dim must be a positive integer")
    sites = []
    for raw in raw_sites:
        values = np.asarray(raw.features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 1 + 2 * covariate_dim:
            raise ValueError("RawSite must have exactly [A,X,A*X] columns")
        a = values[:, 0].copy()
        x = values[:, 1 : 1 + covariate_dim].copy()
        if not np.allclose(values[:, 1 + covariate_dim :], a[:, None] * x, rtol=1e-6, atol=1e-7):
            raise ValueError("RawSite treatment interactions are inconsistent")
        site = CausalSite(x, a, np.asarray(raw.labels, dtype=np.float64).copy(), raw.site_idx)
        _validate_site(site)
        sites.append(site)
    if not sites or len({s.site_idx for s in sites}) != len(sites):
        raise ValueError("R-learner needs uniquely identified sites")
    return sites


def deterministic_folds(num_rows: int, num_folds: int, seed: int) -> np.ndarray:
    """Balanced random folds independent of covariates, treatment and outcome."""
    if num_folds < 2 or num_rows < num_folds:
        raise ValueError("cross-fitting requires at least one row per fold")
    permutation = np.random.default_rng(seed).permutation(num_rows)
    assignments = np.empty(num_rows, dtype=np.int64)
    assignments[permutation] = np.arange(num_rows) % num_folds
    return assignments


@contextmanager
def _timed(stages: Dict[str, object], name: str):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        record = stages.setdefault(name, {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0})
        record["wall_seconds"] += time.perf_counter() - wall
        record["cpu_seconds"] += time.process_time() - cpu
        record["calls"] += 1


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -scores))


def _constant_probability(target: np.ndarray, config: RlearnerConfig) -> float:
    return float((target.sum() + config.constant_pseudocount) / (target.size + 2 * config.constant_pseudocount))


def _local_design(x: np.ndarray, y: np.ndarray) -> Tuple[FederatedSite, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    return FederatedSite(design, y, 0), mean, scale


def _require_nuisance_convergence(report: Dict[str, object]) -> None:
    threshold = float(report["stationarity_threshold"])
    values = [report["kkt_residual_inf"], report["proximal_gradient_mapping_inf"], threshold]
    if (
        not all(np.isfinite(value) for value in values)
        or not report["converged"]
        or threshold <= 0
        or report["kkt_residual_inf"] > threshold
        or report["proximal_gradient_mapping_inf"] > threshold
    ):
        raise RuntimeError("local nuisance logistic LASSO failed stationarity")


def _fit_nuisance_cv(
    x: np.ndarray,
    target: np.ndarray,
    predict_x: np.ndarray,
    config: RlearnerConfig,
    seed: int,
    stages: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Called with outer-training rows only; prediction labels are unavailable."""
    if np.unique(target).size == 1:
        with _timed(stages, "nuisance_refit"):
            probability = _constant_probability(target, config)
            prediction = np.full(predict_x.shape[0], probability, dtype=np.float64)
        return prediction, {
            "fit_kind": "train_only_smoothed_bernoulli_constant",
            "training_rows": int(target.size),
            "pseudocount_per_class": config.constant_pseudocount,
            "probability": probability,
            "selected_lambda": None,
            "optimizer_fits": 0,
            "converged": True,
        }
    folds = deterministic_folds(target.size, config.nuisance_cv_folds, seed)
    # Fixed, declared grid: neither inner-validation nor outer-held-out labels
    # determine a lambda_max or any transformation of the training data.
    lambdas = np.geomspace(
        config.nuisance_lambda_max,
        config.nuisance_lambda_max * config.nuisance_lambda_min_ratio,
        config.nuisance_lambda_count,
    )
    optimizer = OptimizerConfig(max_iter=config.nuisance_max_iter, tolerance=config.nuisance_tolerance)
    losses = np.zeros((config.nuisance_cv_folds, lambdas.size), dtype=np.float64)
    diagnostics = {"optimizer_fits": 0, "optimizer_iterations": 0, "max_stationarity_ratio": 0.0}
    constant_inner_folds = []

    def collect(report):
        _require_nuisance_convergence(report)
        diagnostics["optimizer_fits"] += 1
        diagnostics["optimizer_iterations"] += int(report["iterations"])
        diagnostics["max_stationarity_ratio"] = max(
            diagnostics["max_stationarity_ratio"],
            float(report["kkt_residual_inf"]) / float(report["stationarity_threshold"]),
            float(report["proximal_gradient_mapping_inf"]) / float(report["stationarity_threshold"]),
        )

    with _timed(stages, "nuisance_cv"):
        for fold in range(config.nuisance_cv_folds):
            training = folds != fold
            validation = ~training
            train_y = target[training]
            if np.unique(train_y).size == 1:
                probability = _constant_probability(train_y, config)
                score = math.log(probability / (1 - probability))
                losses[fold, :] = np.sum(np.logaddexp(0, score) - target[validation] * score)
                constant_inner_folds.append(fold)
                continue
            site, mean, scale = _local_design(x[training], train_y)
            validation_design = np.column_stack((np.ones(validation.sum()), (x[validation] - mean) / scale))
            initial = np.zeros(site.design.shape[1])
            initial[0] = math.log(train_y.mean() / (1 - train_y.mean()))
            for idx, lambda_value in enumerate(lambdas):
                initial, report = proximal_gradient_fit(
                    [site], penalty_start=1, lambda_value=float(lambda_value),
                    config=optimizer, initial_coefficients=initial,
                )
                collect(report)
                scores = validation_design.dot(initial)
                losses[fold, idx] = np.sum(np.logaddexp(0, scores) - target[validation] * scores)
        mean_losses = losses.sum(axis=0) / target.size
        minimum = float(mean_losses.min())
        selected = int(np.flatnonzero(mean_losses <= minimum + 1e-12)[0])

    with _timed(stages, "nuisance_refit"):
        site, mean, scale = _local_design(x, target)
        initial = np.zeros(site.design.shape[1])
        initial[0] = math.log(target.mean() / (1 - target.mean()))
        coefficients, report = proximal_gradient_fit(
            [site], penalty_start=1, lambda_value=float(lambdas[selected]),
            config=optimizer, initial_coefficients=initial,
        )
        collect(report)
        prediction = _sigmoid(coefficients[0] + ((predict_x - mean) / scale).dot(coefficients[1:]))
    return prediction, {
        "fit_kind": "site_local_logistic_lasso_nested_cv",
        "training_rows": int(target.size),
        "num_folds": config.nuisance_cv_folds,
        "lambda_path": lambdas.tolist(),
        "lambda_grid_rule": "fixed_absolute_grid_before_seeing_any_rows",
        "validation_mean_logloss": mean_losses.tolist(),
        "selection_rule": "minimum_row_weighted_logloss_ties_1e-12_largest_lambda",
        "selected_index": selected,
        "selected_lambda": float(lambdas[selected]),
        "constant_inner_folds": constant_inner_folds,
        "constant_pseudocount_per_class": config.constant_pseudocount,
        "standardization_scope": "each_inner_training_fold_then_outer_training_refit_only",
        "converged": True,
        **diagnostics,
    }


def crossfit_site_nuisances(
    site: CausalSite,
    config: RlearnerConfig,
    seed: int,
    stages: Dict[str, object] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    _validate_config(config)
    _validate_site(site)
    stages = {} if stages is None else stages
    x = np.asarray(site.covariates, dtype=np.float64)
    y = np.asarray(site.outcome, dtype=np.float64)
    a = np.asarray(site.treatment, dtype=np.float64)
    folds = deterministic_folds(x.shape[0], config.crossfit_folds, seed)
    m_hat = np.full(x.shape[0], np.nan)
    e_hat = np.full(x.shape[0], np.nan)
    reports = []
    for fold in range(config.crossfit_folds):
        training, held_out = folds != fold, folds == fold
        if int(training.sum()) < config.nuisance_cv_folds:
            raise ValueError("outer training fold is too small for nuisance CV")
        inner_seed = seed + 104729 * (fold + 1)
        m_hat[held_out], m_report = _fit_nuisance_cv(x[training], y[training], x[held_out], config, inner_seed, stages)
        e_hat[held_out], e_report = _fit_nuisance_cv(x[training], a[training], x[held_out], config, inner_seed, stages)
        reports.append({
            "fold": fold, "training_rows": int(training.sum()),
            "held_out_rows": int(held_out.sum()), "outcome": m_report, "propensity": e_report,
        })
    if not np.isfinite(m_hat).all() or not np.isfinite(e_hat).all():
        raise RuntimeError("cross-fitting did not produce finite predictions for all rows")
    if not ((0 <= m_hat) & (m_hat <= 1) & (0 <= e_hat) & (e_hat <= 1)).all():
        raise RuntimeError("nuisance predictions are not Bernoulli probabilities")
    clipping_count = int(np.count_nonzero((e_hat < config.propensity_clip) | (e_hat > 1 - config.propensity_clip)))
    e_hat = np.clip(e_hat, config.propensity_clip, 1 - config.propensity_clip)
    return m_hat, e_hat, {
        "crossfit_seed": int(seed),
        "fold_assignment_rule": "balanced_shuffled_rows_independent_of_X_A_Y",
        "fold_assignment_sha256": hashlib.sha256(folds.astype("<i8").tobytes()).hexdigest(),
        "outer_folds": reports,
        "propensity_clip": config.propensity_clip,
        "propensity_clipped_rows": clipping_count,
        "test_rows_used": 0,
    }


def _positive_definite(matrix: np.ndarray, condition_limit: float, name: str) -> float:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("%s must be a finite square matrix" % name)
    if not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-12):
        raise ValueError("%s must be symmetric" % name)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if (
        not np.isfinite(eigenvalues).all()
        or eigenvalues[0] <= 0
        or eigenvalues[-1] / eigenvalues[0] > condition_limit
    ):
        raise ValueError("%s is rank deficient or numerically ill-conditioned" % name)
    return float(eigenvalues[-1] / eigenvalues[0])


def build_site_summary(
    site: CausalSite,
    m_hat: np.ndarray,
    e_hat: np.ndarray,
    config: RlearnerConfig = RlearnerConfig(),
) -> Tuple[SiteSummary, Dict[str, object]]:
    """Local unpenalized residual regression and reference CD precision."""
    _validate_config(config)
    _validate_site(site)
    x = np.asarray(site.covariates, dtype=np.float64)
    m, e = np.asarray(m_hat, dtype=np.float64), np.asarray(e_hat, dtype=np.float64)
    n, d = x.shape[0], x.shape[1] + 1
    if n <= d:
        raise ValueError("local CD summary needs more observations than effect parameters")
    if m.shape != (n,) or e.shape != (n,) or not np.isfinite(m).all() or not np.isfinite(e).all():
        raise ValueError("nuisance predictions must be finite and match local rows")
    if not ((m >= 0) & (m <= 1)).all() or not ((e > 0) & (e < 1)).all():
        raise ValueError("nuisance probabilities are outside their permitted range")
    design = (site.treatment - e)[:, None] * np.column_stack((np.ones(n), x))
    residual_outcome = site.outcome - m
    gram = design.T.dot(design)
    condition = _positive_definite(gram, config.condition_limit, "local residual Gram")
    beta = np.linalg.solve(gram, design.T.dot(residual_outcome))
    residuals = residual_outcome - design.dot(beta)
    stationarity = float(np.max(np.abs(design.T.dot(residuals)) / n))
    stationarity_threshold = config.effect_tolerance * (
        1 + float(np.max(np.abs(design.T.dot(residual_outcome)) / n))
    )
    if not np.isfinite(stationarity) or stationarity > stationarity_threshold:
        raise RuntimeError("local residual regression failed stationarity")
    variance = float(residuals.dot(residuals) / n)
    if not np.isfinite(variance) or variance <= config.residual_variance_floor:
        raise ValueError("local residual variance is degenerate; cannot form CD precision")
    precision = gram / variance
    _positive_definite(precision, config.condition_limit, "local CD precision")
    summary = SiteSummary(int(site.site_idx), int(n), beta, precision)
    return summary, {
        "site_idx": int(site.site_idx), "num_examples": int(n), "effect_dimension": int(d),
        "gram_condition_number": condition,
        "residual_variance": variance, "variance_denominator": "n_not_n_minus_p",
        "precision_rule": "Z_transpose_Z_divided_by_RSS_over_n",
        "local_stationarity_inf": stationarity,
        "local_stationarity_threshold": stationarity_threshold,
        "payload_sha256": hashlib.sha256(summary.payload_bytes()).hexdigest(),
        "payload_bytes": len(summary.payload_bytes()),
    }


def _validate_summaries(summaries: Sequence[SiteSummary], config: RlearnerConfig) -> int:
    if not summaries or any(not isinstance(s, SiteSummary) for s in summaries):
        raise ValueError("effect server accepts SiteSummary objects only")
    d = np.asarray(summaries[0].beta).size
    if d < 2 or len({s.site_idx for s in summaries}) != len(summaries):
        raise ValueError("effect summaries require unique sites and an intercept plus slopes")
    for summary in summaries:
        if (
            isinstance(summary.site_idx, bool)
            or not isinstance(summary.site_idx, (int, np.integer))
            or summary.site_idx < 0
            or isinstance(summary.num_examples, bool)
            or not isinstance(summary.num_examples, (int, np.integer))
            or summary.num_examples <= d
            or np.asarray(summary.beta).shape != (d,)
            or np.asarray(summary.precision).shape != (d, d)
            or not np.isfinite(summary.beta).all()
        ):
            raise ValueError("effect summary identity, count or coefficient shape is invalid")
        _positive_definite(np.asarray(summary.precision), config.condition_limit, "site summary precision")
    return d


def summary_cd_loss(summaries: Sequence[SiteSummary], beta: np.ndarray) -> float:
    """Reference CD quadratic, with its constant, scaled by total sample size."""
    total = sum(summary.num_examples for summary in summaries)
    if total <= 0:
        raise ValueError("CD loss requires nonempty summaries")
    return float(sum((summary.beta - beta).dot(summary.precision).dot(summary.beta - beta) for summary in summaries) / (2 * total))


def solve_shared_quadratic(
    hessian: np.ndarray,
    linear: np.ndarray,
    lambda_value: float,
    penalty_weights: np.ndarray,
    config: RlearnerConfig = RlearnerConfig(),
    initial: np.ndarray = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Server-local coordinate descent; intercept never receives a penalty."""
    _validate_config(config)
    h = np.asarray(hessian, dtype=np.float64)
    b = np.asarray(linear, dtype=np.float64)
    weights = np.asarray(penalty_weights, dtype=np.float64)
    _positive_definite(h, config.condition_limit, "shared CD Hessian")
    if b.shape != (h.shape[0],) or weights.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(weights).all():
        raise ValueError("shared linear term and weights must be finite coefficient vectors")
    if not np.isfinite(lambda_value) or lambda_value < 0 or weights[0] != 0 or np.any(weights[1:] <= 0):
        raise ValueError("L1 weights must leave only the intercept unpenalized")
    if lambda_value == 0:
        beta = np.linalg.solve(h, b)
        iterations = 0
    else:
        beta = np.zeros_like(b) if initial is None else np.asarray(initial, dtype=np.float64).copy()
        if beta.shape != b.shape or not np.isfinite(beta).all():
            raise ValueError("initial shared coefficients are invalid")
        for iterations in range(1, config.effect_max_iter + 1):
            for j in range(beta.size):
                partial = b[j] - h[j].dot(beta) + h[j, j] * beta[j]
                beta[j] = np.sign(partial) * max(abs(partial) - lambda_value * weights[j], 0) / h[j, j]
            gradient = h.dot(beta) - b
            residual = np.maximum(np.abs(gradient) - lambda_value * weights, 0)
            nonzero = beta != 0
            residual[nonzero] = np.abs(gradient[nonzero] + lambda_value * weights[nonzero] * np.sign(beta[nonzero]))
            threshold = config.effect_tolerance * (1 + float(np.max(np.abs(b))))
            if np.max(residual) <= threshold:
                break
        else:
            raise RuntimeError("shared effect LASSO reached iteration limit without stationarity")
    gradient = h.dot(beta) - b
    residual = np.maximum(np.abs(gradient) - lambda_value * weights, 0)
    nonzero = beta != 0
    residual[nonzero] = np.abs(gradient[nonzero] + lambda_value * weights[nonzero] * np.sign(beta[nonzero]))
    kkt = float(np.max(residual))
    threshold = config.effect_tolerance * (1 + float(np.max(np.abs(b))))
    if not np.isfinite(beta).all() or not np.isfinite(kkt) or kkt > threshold:
        raise RuntimeError("shared effect estimate failed stationarity")
    return beta, {
        "iterations": iterations, "converged": True,
        "stop_reason": "closed_form" if lambda_value == 0 else "kkt_stationarity",
        "kkt_residual_inf": kkt, "stationarity_threshold": threshold,
    }


def fit_shared_effect(
    summaries: Sequence[SiteSummary],
    penalty: str,
    config: RlearnerConfig = RlearnerConfig(),
) -> Tuple[LinearProbabilityEffectModel, Dict[str, object]]:
    """The only server entrypoint: accepts summaries, never row-level data."""
    _validate_config(config)
    d = _validate_summaries(summaries, config)
    if penalty not in ("none", "adaptive_lasso"):
        raise ValueError("penalty must be none or adaptive_lasso")
    n = sum(s.num_examples for s in summaries)
    h = sum((np.asarray(s.precision) for s in summaries), np.zeros((d, d))) / n
    b = sum((np.asarray(s.precision).dot(s.beta) for s in summaries), np.zeros(d)) / n
    ordinary_weights = np.r_[0.0, np.ones(d - 1)]
    initial, initial_report = solve_shared_quadratic(h, b, 0.0, ordinary_weights, config)
    if penalty == "none":
        chosen = initial
        path = [{"lambda": 0.0, **initial_report}]
        selected = 0
        weights = ordinary_weights
    else:
        weights = np.r_[0.0, (np.abs(initial[1:]) + config.adaptive_epsilon) ** (-config.adaptive_power)]
        intercept_only = np.zeros(d)
        intercept_only[0] = b[0] / h[0, 0]
        lambda_max = float(np.max(np.abs((h.dot(intercept_only) - b)[1:]) / weights[1:]))
        lambdas = (
            np.geomspace(lambda_max, lambda_max * config.effect_lambda_min_ratio, config.effect_lambda_count)
            if lambda_max > 0 else np.asarray([0.0])
        )
        # Zero is explicitly present so shrinkage is selected only when EBIC
        # supports it. The no-penalty method never searches this path.
        if lambdas[-1] != 0:
            lambdas = np.r_[lambdas, 0.0]
        path, solutions = [], []
        current = intercept_only
        for lambda_value in lambdas:
            current, report = solve_shared_quadratic(h, b, float(lambda_value), weights, config, current)
            active = int(np.count_nonzero(current[1:]))
            # Shared intercept contributes one df regardless of its value.
            degrees = active + 1
            log_combinations = math.lgamma(d) - math.lgamma(active + 1) - math.lgamma(d - active)
            discrepancy = 2 * n * summary_cd_loss(summaries, current)
            ebic = discrepancy + degrees * math.log(n) + 2 * config.ebic_gamma * log_combinations
            path.append({
                "lambda": float(lambda_value), "active_covariates": active,
                "degrees_of_freedom": degrees, "cd_discrepancy": discrepancy,
                "ebic": float(ebic), **report,
            })
            solutions.append(current.copy())
        minimum = min(item["ebic"] for item in path)
        selected = next(i for i, item in enumerate(path) if item["ebic"] <= minimum + 1e-12)
        chosen = solutions[selected]
    selected_report = path[selected]
    model = LinearProbabilityEffectModel(chosen)
    return model, {
        "effect_penalty": penalty, "shared_effect_coefficients": chosen.tolist(),
        "unpenalized_shared_coefficients": initial.tolist(),
        "selected_lambda": selected_report["lambda"], "selected_index": selected,
        "selected_fit": selected_report, "lambda_fits": path,
        "converged": True, "stop_reason": selected_report["stop_reason"],
        "effect_intercept_penalized": False,
        "adaptive_weights": weights.tolist() if penalty == "adaptive_lasso" else None,
        "adaptive_weight_rule": "(abs(unpenalized_shared_beta_j)+epsilon)^(-power)" if penalty == "adaptive_lasso" else None,
        "ebic_rule": "CD_discrepancy+(1+nonzero_slopes)*log(N)+2*gamma*log_choose(p,nonzero_slopes)" if penalty == "adaptive_lasso" else None,
        "selection_rule": "minimum_EBIC_ties_1e-12_largest_lambda" if penalty == "adaptive_lasso" else "no_selection",
        "num_sites": len(summaries), "num_examples": int(n),
        "server_uses_individual_rows": False,
        "additional_sample_size_multiplier": False,
        "fit_fingerprint": hashlib.sha256(np.asarray(chosen, dtype="<f8").tobytes()).hexdigest(),
    }


def fit_homogeneous_rlearner(
    raw_sites: Sequence[RawSite],
    covariate_dim: int,
    penalty: str = "none",
    config: RlearnerConfig = RlearnerConfig(),
    seed: int = 0,
    communication=None,
) -> Tuple[LinearProbabilityEffectModel, Dict[str, object]]:
    """Fit one complete method independently, including all nuisance tuning.

    The runner may wrap this callable with its MethodTimer. The nonoverlapping
    ``stage_timing`` records provide finer detail within that full fit time.
    Optional communication is a measurement.CommunicationLedger; otherwise
    the same logical events are reported directly in the returned metadata.
    """
    _validate_config(config)
    if penalty not in ("none", "adaptive_lasso"):
        raise ValueError("penalty must be none or adaptive_lasso")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    sites = raw_sites_to_causal_sites(raw_sites, covariate_dim)
    stages, summaries, local_reports = {}, [], []
    for site in sites:
        local_wall, local_cpu = time.perf_counter(), time.process_time()
        m_hat, e_hat, nuisance_report = crossfit_site_nuisances(
            site, config, seed + 1000003 * (site.site_idx + 1), stages,
        )
        with _timed(stages, "local_summary"):
            summary, summary_report = build_site_summary(site, m_hat, e_hat, config)
        summaries.append(summary)
        if communication is not None:
            communication.add(
                direction="upload", payload_bytes=len(summary.payload_bytes()),
                messages=1, round_id="r_summary_upload", label="r_local_cd_summary",
            )
        local_reports.append({
            "site_idx": int(site.site_idx), "nuisance": nuisance_report,
            "summary": summary_report,
            "wall_seconds": time.perf_counter() - local_wall,
            "cpu_seconds": time.process_time() - local_cpu,
        })
    with _timed(stages, "server_effect_fit_and_selection"):
        model, report = fit_shared_effect(summaries, penalty, config)
    report.update({
        "method": "homogeneous_r_learner" if penalty == "none" else "homogeneous_r_adaptive_lasso",
        "implementation": "independent_homogeneous_CD_adaptation_of_DF_R_learner",
        "reference_doi": "10.1093/biomtc/ujag034",
        "reference_code": "https://github.com/Jin-hongLi/DF-R-learner/blob/main/DFRlearn_main_c1n100.R",
        "effect_scale": "probability_difference",
        "prediction_clipping": False,
        "working_model": "shared_linear_probability_effect",
        "covariance_model": "reference_homoskedastic_residual_working_covariance",
        "limitations": [
            "binary_DGP_probability_effect_need_not_be_linear",
            "outcome_dependent_site_partition_need_not_preserve_homogeneous_causal_effect",
            "reference_working_covariance_is_not_a_heteroskedastic_inference_guarantee",
        ],
        "configuration": asdict(config), "seed": seed,
        "local_reports": local_reports, "stage_timing": stages,
        "full_fit_wall_seconds": time.perf_counter() - started_wall,
        "full_fit_cpu_seconds": time.process_time() - started_cpu,
        "timing_scope": "serial_simulation_all_local_nested_selection_refits_summaries_and_server_selection",
        "communication": {
            "synchronization_rounds": 1,
            "upload_messages": len(summaries), "download_messages": 0,
            "upload_bytes": sum(len(s.payload_bytes()) for s in summaries),
            "download_bytes": 0,
            "wire_encoding": "per_site_int64_site_id_and_n_float64_beta_dense_precision",
            "network_time_measured": False,
            "evaluation_distribution_included": False,
            "payloads": [local["summary"]["payload_sha256"] for local in local_reports],
        },
    })
    return model, report


def validate_fit_report(report, config, penalty, coefficients) -> None:
    """Validate saved scientific evidence without fitting or reading site rows.

    ``config`` is the manifest's RlearnerConfig or its JSON dictionary. The
    caller separately validates artifact hashes, the run identity and provenance.
    This rejects internally inconsistent successful flags, stationarity reports,
    selection metadata, and communication/timing evidence on resume.
    """
    try:
        if isinstance(config, dict):
            config = RlearnerConfig(**config)
        _validate_config(config)
        if penalty not in ("none", "adaptive_lasso"):
            raise ValueError("invalid manifest R penalty")
        beta = np.asarray(coefficients, dtype=np.float64)
        d = beta.size
        if beta.shape != (d,) or d < 2 or not np.isfinite(beta).all():
            raise ValueError("invalid saved R coefficients")
        if report["configuration"] != asdict(config) or report["effect_penalty"] != penalty:
            raise ValueError("R fit configuration differs from manifest")
        expected_method = "homogeneous_r_learner" if penalty == "none" else "homogeneous_r_adaptive_lasso"
        if (
            report["method"] != expected_method or report["converged"] is not True
            or report["effect_scale"] != "probability_difference"
            or report["prediction_clipping"] is not False
            or report["effect_intercept_penalized"] is not False
            or report["server_uses_individual_rows"] is not False
            or report["additional_sample_size_multiplier"] is not False
        ):
            raise ValueError("R method identity or scientific completion is inconsistent")
        if not np.array_equal(np.asarray(report["shared_effect_coefficients"]), beta):
            raise ValueError("R fit coefficients differ from model artifact")
        if report["fit_fingerprint"] != hashlib.sha256(np.asarray(beta, dtype="<f8").tobytes()).hexdigest():
            raise ValueError("R coefficient fingerprint mismatch")

        def integer(value, name, minimum=0):
            if type(value) is not int or value < minimum:
                raise ValueError("invalid R %s" % name)
            return value

        def finite(value, name, minimum=0.0):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
                raise ValueError("invalid R %s" % name)
            return value

        def checksum(value):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("invalid R summary/fold fingerprint")

        n = integer(report["num_examples"], "sample size", d + 1)
        path = report["lambda_fits"]
        selected = integer(report["selected_index"], "selected lambda index")
        if not isinstance(path, list) or not path or selected >= len(path):
            raise ValueError("invalid R lambda path")
        if report["selected_fit"] != path[selected] or report["selected_lambda"] != path[selected]["lambda"]:
            raise ValueError("R selected fit differs from its lambda path")
        if report["stop_reason"] != path[selected]["stop_reason"]:
            raise ValueError("R terminal reason differs from selected fit")
        for item in path:
            lam = finite(item["lambda"], "lambda")
            threshold = finite(item["stationarity_threshold"], "stationarity threshold", config.effect_tolerance)
            residual = finite(item["kkt_residual_inf"], "KKT residual")
            iterations = integer(item["iterations"], "effect iterations")
            if item["converged"] is not True or residual > threshold or iterations > config.effect_max_iter:
                raise ValueError("R effect candidate failed stationarity")
            if item["stop_reason"] != ("closed_form" if lam == 0 else "kkt_stationarity") or (iterations == 0) != (lam == 0):
                raise ValueError("R effect optimizer terminal evidence is inconsistent")
        initial = np.asarray(report["unpenalized_shared_coefficients"], dtype=np.float64)
        if initial.shape != beta.shape or not np.isfinite(initial).all():
            raise ValueError("invalid R unpenalized shared initialization")
        if penalty == "none":
            if len(path) != 1 or selected != 0 or path[0]["lambda"] != 0 or report["adaptive_weights"] is not None:
                raise ValueError("unpenalized R fit unexpectedly selected or penalized")
            if not np.array_equal(initial, beta):
                raise ValueError("unpenalized R solution differs from initialization")
        else:
            weights = np.r_[0.0, (np.abs(initial[1:]) + config.adaptive_epsilon) ** (-config.adaptive_power)]
            if not np.allclose(np.asarray(report["adaptive_weights"]), weights, rtol=1e-13, atol=0):
                raise ValueError("R adaptive weights differ from their declared rule")
            lambdas = [item["lambda"] for item in path]
            expected_path = (
                np.r_[np.geomspace(lambdas[0], lambdas[0] * config.effect_lambda_min_ratio, config.effect_lambda_count), 0.0]
                if lambdas[0] > 0 else np.asarray([0.0])
            )
            if len(lambdas) != len(expected_path) or not np.allclose(lambdas, expected_path, rtol=1e-13, atol=0):
                raise ValueError("R effect lambda grid differs from manifest")
            for item in path:
                active = integer(item["active_covariates"], "active covariates")
                if active > d - 1 or item["degrees_of_freedom"] != 1 + active:
                    raise ValueError("R EBIC does not count shared coefficients correctly")
                discrepancy = finite(item["cd_discrepancy"], "CD discrepancy")
                combin = math.lgamma(d) - math.lgamma(active + 1) - math.lgamma(d - active)
                expected = discrepancy + (active + 1) * math.log(n) + 2 * config.ebic_gamma * combin
                if not math.isclose(finite(item["ebic"], "EBIC"), expected, rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("R EBIC is inconsistent with candidate evidence")
            minimum = min(item["ebic"] for item in path)
            expected_selected = next(i for i, item in enumerate(path) if item["ebic"] <= minimum + 1e-12)
            if selected != expected_selected or path[selected]["active_covariates"] != int(np.count_nonzero(beta[1:])):
                raise ValueError("R selected candidate violates EBIC or sparsity evidence")

        def nuisance(item, training_rows):
            if item["training_rows"] != training_rows or item["converged"] is not True:
                raise ValueError("R nuisance training scope or convergence is invalid")
            fits = integer(item["optimizer_fits"], "nuisance optimizer count")
            if item["fit_kind"] == "train_only_smoothed_bernoulli_constant":
                prior = config.constant_pseudocount
                probability = finite(item["probability"], "constant probability")
                expected = [prior / (training_rows + 2 * prior), (training_rows + prior) / (training_rows + 2 * prior)]
                if fits != 0 or item["selected_lambda"] is not None or item["pseudocount_per_class"] != prior or not any(math.isclose(probability, p, rel_tol=1e-14) for p in expected):
                    raise ValueError("R constant nuisance violates train-only smoothing rule")
                return
            if item["fit_kind"] != "site_local_logistic_lasso_nested_cv" or item["num_folds"] != config.nuisance_cv_folds:
                raise ValueError("R nuisance did not use the declared nested CV")
            ratio = finite(item["max_stationarity_ratio"], "nuisance stationarity ratio")
            if ratio > 1:
                raise ValueError("R nuisance candidate failed stationarity")
            iterations = integer(item["optimizer_iterations"], "nuisance iterations", fits)
            constant_folds = item["constant_inner_folds"]
            if not isinstance(constant_folds, list) or any(type(f) is not int or not 0 <= f < config.nuisance_cv_folds for f in constant_folds) or len(set(constant_folds)) != len(constant_folds):
                raise ValueError("invalid R inner constant fold evidence")
            if fits != (config.nuisance_cv_folds - len(constant_folds)) * config.nuisance_lambda_count + 1 or iterations > fits * config.nuisance_max_iter:
                raise ValueError("R nuisance candidate inventory is incomplete")
            lambdas = np.geomspace(config.nuisance_lambda_max, config.nuisance_lambda_max * config.nuisance_lambda_min_ratio, config.nuisance_lambda_count)
            values = np.asarray(item["validation_mean_logloss"], dtype=np.float64)
            if values.shape != lambdas.shape or not np.isfinite(values).all() or np.any(values < 0) or not np.array_equal(item["lambda_path"], lambdas):
                raise ValueError("invalid R nuisance validation/grid evidence")
            selection = int(np.flatnonzero(values <= values.min() + 1e-12)[0])
            if item["selected_index"] != selection or item["selected_lambda"] != lambdas[selection]:
                raise ValueError("R nuisance did not select the declared minimum validation loss")

        sites = report["local_reports"]
        if len(sites) != integer(report["num_sites"], "site count", 1) or len({s["site_idx"] for s in sites}) != len(sites):
            raise ValueError("invalid R local site inventory")
        total_count, fingerprints = 0, []
        seed = integer(report["seed"], "seed")
        payload_size = 16 + 8 * (d + d * d)
        for site in sites:
            idx = integer(site["site_idx"], "site index")
            summary = site["summary"]
            count = integer(summary["num_examples"], "local sample size", d + 1)
            total_count += count
            if summary["site_idx"] != idx or summary["effect_dimension"] != d or summary["payload_bytes"] != payload_size:
                raise ValueError("R summary dimension, count or encoding mismatch")
            if not config.residual_variance_floor < finite(summary["residual_variance"], "local residual variance") or not 1 <= finite(summary["gram_condition_number"], "local condition") <= config.condition_limit:
                raise ValueError("R local covariance evidence is degenerate")
            if summary["precision_rule"] != "Z_transpose_Z_divided_by_RSS_over_n" or summary["variance_denominator"] != "n_not_n_minus_p":
                raise ValueError("R precision scale differs from reference")
            if finite(summary["local_stationarity_inf"], "local stationarity") > finite(summary["local_stationarity_threshold"], "local stationarity threshold", config.effect_tolerance):
                raise ValueError("R local residual estimate failed stationarity")
            checksum(summary["payload_sha256"])
            fingerprints.append(summary["payload_sha256"])
            crossfit = site["nuisance"]
            if crossfit["crossfit_seed"] != seed + 1000003 * (idx + 1) or crossfit["test_rows_used"] != 0 or crossfit["propensity_clip"] != config.propensity_clip:
                raise ValueError("R nuisance cross-fit scope differs from manifest")
            checksum(crossfit["fold_assignment_sha256"])
            if not 0 <= integer(crossfit["propensity_clipped_rows"], "propensity clipping count") <= count:
                raise ValueError("invalid R propensity clipping count")
            outer = crossfit["outer_folds"]
            if len(outer) != config.crossfit_folds or [f["fold"] for f in outer] != list(range(config.crossfit_folds)):
                raise ValueError("incomplete R cross-fit folds")
            if sum(integer(f["held_out_rows"], "held-out rows", 1) for f in outer) != count:
                raise ValueError("R cross-fitting does not cover all local rows")
            for fold in outer:
                training = integer(fold["training_rows"], "outer training rows", 1)
                if training + fold["held_out_rows"] != count:
                    raise ValueError("R outer fold sample counts are inconsistent")
                nuisance(fold["outcome"], training)
                nuisance(fold["propensity"], training)
            for key in ("wall_seconds", "cpu_seconds"):
                finite(site[key], "local " + key)
        if total_count != n:
            raise ValueError("R total sample size differs from site summaries")
        comm = report["communication"]
        expected_comm = dict(synchronization_rounds=1, upload_messages=len(sites), download_messages=0,
                             upload_bytes=payload_size * len(sites), download_bytes=0, payloads=fingerprints)
        if any(comm[key] != value for key, value in expected_comm.items()):
            raise ValueError("R one-shot summary communication evidence is inconsistent")
        for key in ("synchronization_rounds", "upload_messages", "download_messages", "upload_bytes", "download_bytes"):
            integer(comm[key], key)
        stages = report["stage_timing"]
        if not {"nuisance_refit", "local_summary", "server_effect_fit_and_selection"}.issubset(stages) or not set(stages).issubset({"nuisance_cv", "nuisance_refit", "local_summary", "server_effect_fit_and_selection"}):
            raise ValueError("R detailed timing stages are missing or unknown")
        for clock in ("wall", "cpu"):
            total = finite(report["full_fit_" + clock + "_seconds"], "full fit " + clock)
            stage_sum = 0.0
            for stage in stages.values():
                integer(stage["calls"], "timing calls", 1)
                stage_sum += finite(stage[clock + "_seconds"], "stage " + clock)
            if stage_sum > total + 1e-6:
                raise ValueError("R stage timing double-counts the full fit")
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("malformed R fit report") from error
