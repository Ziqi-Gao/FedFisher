"""Federated interaction-LASSO adaptation for survival pseudo-observations.

This Python benchmark is not official R/DataSHIELD dsLassoCov. Site loss and
gradient summaries define nonlinear least squares, not a Bernoulli likelihood.
The optimizer checks first-order stationarity; the objective is nonconvex and
no global optimum is claimed. Historical logistic code is deliberately intact.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np

from .links import survival_mean, survival_mean_derivative
from .pseudo import site_pseudo_response


@dataclass(frozen=True)
class DslassoConfig:
    num_folds: int = 5
    num_lambdas: int = 30
    lambda_min_ratio: float = .001
    max_iter: int = 4000
    tolerance: float = 1e-6
    initial_step: float = 1.0
    min_step: float = 1e-12
    scale_floor: float = 1e-12

    def validate(self):
        for key in ("num_folds", "num_lambdas", "max_iter"):
            value = getattr(self, key)
            if type(value) is not int or value < (2 if key == "num_folds" else 1):
                raise ValueError("invalid " + key)
        for key in ("tolerance", "initial_step", "min_step", "scale_floor"):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("invalid " + key)
        if not 0 < self.lambda_min_ratio <= 1 or self.min_step > self.initial_step:
            raise ValueError("invalid lambda range or optimizer step")


@dataclass(frozen=True)
class SurvivalLinearModel:
    intercept: float
    coefficients: np.ndarray

    def predict_eta(self, design):
        design = np.asarray(design, dtype=np.float64)
        if (design.ndim != 2 or design.shape[1] != len(self.coefficients)
                or not np.isfinite(design).all()):
            raise ValueError("invalid prediction design")
        result = float(self.intercept) + design @ self.coefficients
        if not np.isfinite(result).all():
            raise ValueError("nonfinite prediction")
        return result


class _Communication:
    """Numeric summary accounting; excludes serialization and actual transport."""

    def __init__(self):
        self.phases = {}

    def record(self, kind, num_sites, upload, download):
        entry = self.phases.setdefault(kind, dict(rounds=0, upload_bytes=0,
                                                  download_bytes=0, site_messages=0))
        entry["rounds"] += 1
        entry["site_messages"] += num_sites
        entry["upload_bytes"] += int(upload)
        entry["download_bytes"] += int(download)

    def report(self):
        return dict(phases=self.phases,
                    aggregation_rounds=sum(v["rounds"] for v in self.phases.values()),
                    upload_bytes=sum(v["upload_bytes"] for v in self.phases.values()),
                    download_bytes=sum(v["download_bytes"] for v in self.phases.values()),
                    accounting="float64_numeric_payload_only_no_transport_or_serialization",
                    one_shot=False)


def aggregate_loss_gradient(sites, beta, communication=None):
    """Aggregate sums only; ``sites`` contain (design including intercept, PV)."""
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim != 1 or not len(beta) or not np.isfinite(beta).all() or not sites:
        raise ValueError("invalid coefficients or empty sites")
    loss_sum = 0.
    gradient_sum = np.zeros_like(beta)
    total = 0
    for design, response in sites:
        design = np.asarray(design, dtype=np.float64)
        response = np.asarray(response, dtype=np.float64)
        if (design.ndim != 2 or design.shape[1] != len(beta) or not len(design)
                or response.shape != (len(design),) or not np.isfinite(design).all()
                or not np.isfinite(response).all()):
            raise ValueError("invalid local design/pseudo-response")
        eta = design @ beta
        residual = survival_mean(eta) - response
        loss_sum += .5 * float(residual @ residual)
        gradient_sum += design.T @ (residual * survival_mean_derivative(eta))
        total += len(design)
    if communication is not None:
        n = len(sites)
        communication.record("loss_gradient", n, n * 8 * (len(beta) + 2),
                             n * 8 * len(beta))
    return loss_sum / total, gradient_sum / total


def _residuals(beta, gradient, penalty, penalty_start, step):
    proximal = beta - step * gradient
    tail = proximal[penalty_start:]
    proximal[penalty_start:] = np.sign(tail) * np.maximum(np.abs(tail) - step * penalty, 0)
    residual = np.abs(gradient).copy()
    tail_beta = beta[penalty_start:]
    tail_gradient = gradient[penalty_start:]
    residual[penalty_start:] = np.where(
        tail_beta != 0, np.abs(tail_gradient + penalty * np.sign(tail_beta)),
        np.maximum(np.abs(tail_gradient) - penalty, 0))
    return float(np.max(residual)), float(np.max(np.abs((beta - proximal) / step)))


def proximal_fit(sites, initial, penalty_start, penalty, config, communication=None):
    """Monotone proximal gradient with checked KKT and gradient mapping."""
    config.validate()
    beta = np.asarray(initial, dtype=np.float64).copy()
    if (beta.ndim != 1 or not len(beta) or not np.isfinite(beta).all()
            or type(penalty_start) is not int or not 0 <= penalty_start <= len(beta)
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError("invalid proximal fit inputs")
    step = config.initial_step
    smooth, gradient = aggregate_loss_gradient(sites, beta, communication)
    initial_objective = smooth + penalty * np.abs(beta[penalty_start:]).sum()
    backtracks = 0
    converged = False
    for iteration in range(config.max_iter + 1):
        kkt, mapping = _residuals(beta, gradient, penalty, penalty_start, step)
        threshold = config.tolerance * (1 + np.max(np.abs(beta)))
        if kkt <= threshold and mapping <= threshold:
            converged = True
            break
        if iteration == config.max_iter:
            break
        trial = min(config.initial_step, step * 2)
        while True:
            candidate = beta - trial * gradient
            tail = candidate[penalty_start:]
            candidate[penalty_start:] = np.sign(tail) * np.maximum(np.abs(tail) - trial * penalty, 0)
            candidate_smooth, candidate_gradient = aggregate_loss_gradient(sites, candidate, communication)
            delta = candidate - beta
            bound = smooth + gradient @ delta + (delta @ delta) / (2 * trial)
            if candidate_smooth <= bound + 1e-14:
                break
            trial *= .5
            backtracks += 1
            if trial < config.min_step:
                raise RuntimeError("survival LASSO line search exhausted min_step")
        step = trial
        beta, smooth, gradient = candidate, candidate_smooth, candidate_gradient
    report = dict(iterations=int(iteration), converged=bool(converged),
                  convergence_target="first_order_stationarity_not_global_optimum",
                  initial_objective=float(initial_objective),
                  final_objective=float(smooth + penalty * np.abs(beta[penalty_start:]).sum()),
                  kkt_residual_inf=kkt, proximal_gradient_mapping_inf=mapping,
                  stationarity_threshold=float(threshold), final_step_size=float(step),
                  backtracks=backtracks)
    return beta, report


def _require_stationary(report):
    if not report["converged"]:
        raise RuntimeError("survival LASSO exhausted its iteration budget before stationarity")


def _prepare(rows, tau, config, communication):
    # Responses are formed independently inside the current CV partition/site.
    raw = [(np.asarray(r.design, dtype=np.float64), site_pseudo_response(r, tau)) for r in rows]
    count = sum(len(x) for x, _ in raw)
    width = raw[0][0].shape[1]
    means = sum((x.sum(axis=0) for x, _ in raw), np.zeros(width)) / count
    squares = sum((np.square(x).sum(axis=0) for x, _ in raw), np.zeros(width)) / count
    scales = np.sqrt(np.maximum(squares - means ** 2, 0))
    scales[scales < config.scale_floor] = 1
    communication.record("standardization", len(raw), len(raw) * (8 + 16 * width),
                         len(raw) * 16 * width)
    prepared = [(np.column_stack((np.ones(len(x)), (x - means) / scales)), y)
                for x, y in raw]
    response_sum = sum(float(y.sum()) for _, y in raw)
    communication.record("response_initialization", len(raw), len(raw) * 16, len(raw) * 8)
    mean_response = np.clip(response_sum / count, .05, .95)
    initial = np.zeros(width + 1)
    initial[0] = np.log(-np.log(mean_response))
    return prepared, means, scales, initial


def _path(prepared, initial, p, fractions, config, communication):
    penalty_start = p + 2  # [intercept, A, X..., AX...]
    reduced = [(x[:, :penalty_start], y) for x, y in prepared]
    start, start_report = proximal_fit(reduced, initial[:penalty_start],
                                       penalty_start, 0., config, communication)
    _require_stationary(start_report)
    beta = np.zeros_like(initial)
    beta[:penalty_start] = start
    _, gradient = aggregate_loss_gradient(prepared, beta, communication)
    lambda_max = float(np.max(np.abs(gradient[penalty_start:])))
    solutions, reports = [], []
    for fraction in fractions:
        beta, report = proximal_fit(prepared, beta, penalty_start,
                                    lambda_max * fraction, config, communication)
        _require_stationary(report)
        solutions.append(beta.copy())
        reports.append(report)
    return solutions, dict(lambda_max=lambda_max, unpenalized_start=start_report, fits=reports)


def fit_dslasso(sites, tau, seed=0, config=DslassoConfig()):
    """Training-only site CV; each validation partition computes its own PVs.

    CV selects a common lambda *fraction*: each training fold obtains its own
    lambda_max without validation outcomes. The final refit uses the full
    training lambda_max. Neither test rows nor pooled survival curves are used.
    """
    started = time.perf_counter()
    config.validate()
    if not sites or type(seed) is not int or seed < 0 or not np.isfinite(tau) or tau <= 0:
        raise ValueError("invalid sites, seed or tau")
    rows = [site.rows for site in sites]
    p = rows[0].x.shape[1]
    ids = [site.site_id for site in sites]
    if len(set(ids)) != len(ids) or any(r.x.shape[1] != p for r in rows):
        raise ValueError("duplicate site identifiers or inconsistent dimensions")
    if any(len(r.x) < 2 * config.num_folds for r in rows):
        raise ValueError("at least two validation observations per site/fold required")
    fractions = np.geomspace(1., config.lambda_min_ratio, config.num_lambdas)
    assignments = []
    for site in sites:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(site.site_id), 28461]))
        labels = np.empty(len(site.rows.x), dtype=int)
        labels[rng.permutation(len(labels))] = np.arange(len(labels)) % config.num_folds
        assignments.append(labels)
    communication = _Communication()
    squared_error = np.zeros(config.num_lambdas)
    validation_n = 0
    cv_paths = []
    for fold in range(config.num_folds):
        training = [r.subset(np.flatnonzero(a != fold)) for r, a in zip(rows, assignments)]
        validation = [r.subset(np.flatnonzero(a == fold)) for r, a in zip(rows, assignments)]
        prepared, means, scales, initial = _prepare(training, tau, config, communication)
        solutions, report = _path(prepared, initial, p, fractions, config, communication)
        report["fold"] = fold
        cv_paths.append(report)
        for r in validation:
            design = np.column_stack((np.ones(len(r.x)), (r.design - means) / scales))
            target = site_pseudo_response(r, tau)
            for index, beta in enumerate(solutions):
                residual = survival_mean(design @ beta) - target
                squared_error[index] += float(residual @ residual)
            validation_n += len(r.x)
        communication.record("validation_path_loss", len(sites),
                             len(sites) * 8 * (1 + config.num_lambdas),
                             len(sites) * 8 * (2 + 2 * p) * config.num_lambdas)
    cv_loss = squared_error / validation_n
    # Fractions descend: np.argmin chooses the stronger penalty in exact ties.
    selected = int(np.argmin(cv_loss))
    cv_seconds = time.perf_counter() - started
    refit_started = time.perf_counter()
    prepared, means, scales, initial = _prepare(rows, tau, config, communication)
    solutions, refit = _path(prepared, initial, p, fractions[:selected + 1], config, communication)
    beta = solutions[-1]
    coefficients = beta[1:] / scales
    model = SurvivalLinearModel(float(beta[0] - means @ coefficients), coefficients)
    diagnostics = dict(method="dslassocov_style_survival_nls_lasso", config=asdict(config),
                       objective="half_mean_squared_error_plus_interaction_l1",
                       link="log_minus_log_survival", pseudo_scope="site_and_cv_partition_local",
                       selected_lambda_index=selected,
                       selected_lambda_fraction=float(fractions[selected]),
                       selected_lambda_value=float(refit["lambda_max"] * fractions[selected]),
                       lambda_fractions=fractions.tolist(), cv_loss=cv_loss.tolist(),
                       cv_paths=cv_paths, refit_start=refit["unpenalized_start"],
                       refit_path=refit["fits"], refit_lambda_max=refit["lambda_max"],
                       final_fit=refit["fits"][-1], communication=communication.report(),
                       site_sizes=[len(r.x) for r in rows],
                       timing_seconds=dict(cross_validation=cv_seconds,
                                           refit=time.perf_counter() - refit_started,
                                           full_fit=time.perf_counter() - started))
    return model, diagnostics


def validate_fit_diagnostics(diagnostics, config, arrays, site_sizes):
    """Pure saved-evidence checks, without fitting or prediction."""
    expected = asdict(config) if isinstance(config, DslassoConfig) else dict(config)
    cfg = DslassoConfig(**expected)
    cfg.validate()
    d = diagnostics
    if (d.get("config") != expected or d.get("method") != "dslassocov_style_survival_nls_lasso"
            or d.get("objective") != "half_mean_squared_error_plus_interaction_l1"
            or d.get("link") != "log_minus_log_survival"
            or d.get("pseudo_scope") != "site_and_cv_partition_local"
            or d.get("site_sizes") != list(site_sizes)):
        raise ValueError("survival LASSO fit identity mismatch")
    coefficients = np.asarray(arrays["coefficients"])
    intercept = np.asarray(arrays["intercept"])
    if (coefficients.ndim != 1 or len(coefficients) < 5 or len(coefficients) % 2 != 1
            or intercept.size != 1 or not np.isfinite(coefficients).all()
            or not np.isfinite(intercept).all()):
        raise ValueError("invalid saved linear model")
    fractions = np.geomspace(1., cfg.lambda_min_ratio, cfg.num_lambdas)
    losses = np.asarray(d["cv_loss"], dtype=float)
    selected = d["selected_lambda_index"]
    if (losses.shape != fractions.shape or not np.isfinite(losses).all() or np.any(losses < 0)
            or type(selected) is not int or selected != int(np.argmin(losses))
            or not np.array_equal(d["lambda_fractions"], fractions)
            or d["selected_lambda_fraction"] != fractions[selected]
            or not np.isfinite(d["refit_lambda_max"]) or d["refit_lambda_max"] < 0
            or d["selected_lambda_value"] != d["refit_lambda_max"] * fractions[selected]):
        raise ValueError("invalid CV selection")
    if len(d["cv_paths"]) != cfg.num_folds or len(d["refit_path"]) != selected + 1:
        raise ValueError("incomplete optimizer paths")
    reports = [d["refit_start"]] + d["refit_path"]
    for fold, path in enumerate(d["cv_paths"]):
        if (path["fold"] != fold or len(path["fits"]) != cfg.num_lambdas
                or not np.isfinite(path["lambda_max"]) or path["lambda_max"] < 0):
            raise ValueError("invalid CV path")
        reports.extend([path["unpenalized_start"]] + path["fits"])
    for report in reports:
        if (report["converged"] is not True or type(report["iterations"]) is not int
                or not 0 <= report["iterations"] <= cfg.max_iter
                or report["convergence_target"] != "first_order_stationarity_not_global_optimum"):
            raise ValueError("unconverged optimizer result")
        for key in ("kkt_residual_inf", "proximal_gradient_mapping_inf", "stationarity_threshold",
                    "initial_objective", "final_objective", "final_step_size"):
            if not np.isfinite(report[key]) or report[key] < 0:
                raise ValueError("nonfinite or negative optimizer evidence")
        if (report["stationarity_threshold"] <= 0 or report["final_step_size"] < cfg.min_step
                or report["kkt_residual_inf"] > report["stationarity_threshold"]
                or report["proximal_gradient_mapping_inf"] > report["stationarity_threshold"]
                or report["final_objective"] > report["initial_objective"] + 1e-10):
            raise ValueError("invalid stationarity evidence")
    if d["final_fit"] != d["refit_path"][-1]:
        raise ValueError("final fit does not match refit path")
    communication = d["communication"]
    if communication["one_shot"] is not False or communication["accounting"] != (
            "float64_numeric_payload_only_no_transport_or_serialization"):
        raise ValueError("invalid communication contract")
    phases = communication["phases"]
    if set(phases) != {"standardization", "response_initialization", "loss_gradient", "validation_path_loss"}:
        raise ValueError("incomplete communication phases")
    for phase in phases.values():
        for key in ("rounds", "site_messages", "upload_bytes", "download_bytes"):
            if type(phase[key]) is not int or phase[key] < 0:
                raise ValueError("invalid communication counts")
        if phase["site_messages"] != phase["rounds"] * len(site_sizes):
            raise ValueError("incorrect site message count")
    for output, field in (("aggregation_rounds", "rounds"), ("upload_bytes", "upload_bytes"),
                          ("download_bytes", "download_bytes")):
        if communication[output] != sum(phase[field] for phase in phases.values()):
            raise ValueError("communication totals mismatch")
    times = d["timing_seconds"]
    if set(times) != {"cross_validation", "refit", "full_fit"} or any(
            not np.isfinite(v) or v < 0 for v in times.values()):
        raise ValueError("invalid fit timings")
    if times["cross_validation"] + times["refit"] > times["full_fit"] + 1e-6:
        raise ValueError("timing components exceed full fit")
