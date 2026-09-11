"""Training-only compression and holdout scoring summaries for FedFisher-EM.

The covariance range finder streams parameter gradients and stores D-by-(r+k),
never D-by-D. Complete gradients used for diagnostic directional derivatives
are evaluator-only and are not uploaded.
"""

import math
from time import perf_counter, process_time

import numpy as np

from .fisher_models import (array_hash, model_from_vector, predict_tau,
                           tau_and_gradient, tau_and_directional_derivatives)
from .interventions import make_permutations, perturbed_covariates, validate_permutations


def score_metric(fisher, config):
    if config.compression == "euclidean_gradient":
        return np.ones_like(fisher)
    return fisher + config.score_damping


def _canonical_signs(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    for index in range(matrix.shape[1]):
        pivot = int(np.argmax(np.abs(matrix[:, index])))
        if matrix[pivot, index] < 0:
            matrix[:, index] *= -1
    return matrix


def build_basis(local, config):
    started_wall, started_cpu = perf_counter(), process_time()
    dimension, rank = local.theta.size, config.rank
    if rank > dimension:
        raise ValueError("compression rank exceeds parameter dimension")
    metric = score_metric(local.fisher, config)
    if not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError("the scoring metric must be strictly positive")
    inverse_sqrt = 1. / np.sqrt(metric)
    direction_seed = config.direction_seed_offset + local.seed
    rng = np.random.default_rng(direction_seed)
    selected = np.random.default_rng(config.basis_seed_offset + local.seed + local.site_id * 100003).permutation(len(local.train))[:config.basis_max_rows]
    x = local.train.x[selected]
    plan = make_permutations(len(x), x.shape[1], config.basis_seed_offset + local.seed + local.site_id * 100003 + 1, config.basis_repeats)

    def gradients():
        # Each variable has exactly the same number of perturbation examples.
        for row in x:
            yield tau_and_gradient(local.model, row)[1] * inverse_sqrt
        for _, _, changed in perturbed_covariates(x, plan):
            for row in changed:
                yield tau_and_gradient(local.model, row)[1] * inverse_sqrt

    gradient_count = len(x) * (1 + x.shape[1] * config.basis_repeats)
    if config.compression == "random_orthogonal":
        basis = _canonical_signs(np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")[0])
        # No data-dependent selection; energy below is an explicitly costed audit.
        total_energy, captured_energy = 0., 0.
        for gradient in gradients():
            total_energy += float(gradient @ gradient)
            captured_energy += float(np.square(gradient @ basis).sum())
        eigenvalues = None
        passes = 1
    else:
        width = min(dimension, rank + config.range_oversampling)
        omega = rng.normal(size=(dimension, width)) / math.sqrt(width)
        projected_covariance = np.zeros((dimension, width), dtype=np.float64)
        for gradient in gradients():
            projected_covariance += gradient[:, None] * (gradient @ omega)[None, :] / gradient_count
        span = np.linalg.qr(projected_covariance, mode="reduced")[0]
        rayleigh = np.zeros((width, width), dtype=np.float64)
        total_energy = 0.
        for gradient in gradients():
            projected = gradient @ span
            rayleigh += np.outer(projected, projected) / gradient_count
            total_energy += float(gradient @ gradient)
        values, directions = np.linalg.eigh((rayleigh + rayleigh.T) / 2)
        order = np.argsort(-values, kind="stable")[:rank]
        basis = _canonical_signs(span @ directions[:, order])
        eigenvalues = values[order].tolist()
        captured_energy = float(values[order].sum()) * gradient_count
        passes = 2
    if not np.isfinite(basis).all() or not np.allclose(basis.T @ basis, np.eye(rank), rtol=0, atol=1e-9):
        raise FloatingPointError("compression basis is not finite and orthonormal")
    if captured_energy > total_energy + 1e-9 * max(1., total_energy) or captured_energy < -1e-9 * max(1., total_energy):
        raise FloatingPointError("invalid randomized range-finder energy diagnostic")
    residual_fraction = max(0., total_energy - captured_energy) / total_energy if total_energy > 0 else 0.
    report = {
        "compression": config.compression,
        "metric": "identity" if config.compression == "euclidean_gradient" else "local_mean_Fisher_plus_score_damping",
        "score_damping": config.score_damping,
        "aggregation_damping": 0.,
        "rank": rank,
        "direction_seed": direction_seed,
        "basis_sample_seed": config.basis_seed_offset + local.seed + local.site_id * 100003,
        "basis_training_rows": len(x), "basis_max_rows": config.basis_max_rows,
        "basis_training_row_ids_sha256": array_hash(local.train.row_ids[selected]),
        "candidate_variables": x.shape[1], "perturbation_repeats_per_variable": config.basis_repeats,
        "gradient_examples_per_pass": gradient_count, "gradient_passes": passes,
        "range_oversampling": config.range_oversampling,
        "construction": "fixed_seed_random_QR" if config.compression == "random_orthogonal" else "two_pass_streamed_randomized_covariance_range_finder",
        "matrix_free": True, "parameter_square_matrix_allocated": False,
        "rayleigh_eigenvalues": eigenvalues,
        "observed_gradient_energy": total_energy / gradient_count,
        "observed_captured_energy": captured_energy / gradient_count,
        "observed_relative_projection_energy_error": residual_fraction,
        "error_scope": "empirical_capped_training_gradient_collection_not_population_or_spectral_bound",
        "wall_seconds": perf_counter() - started_wall,
        "cpu_seconds": process_time() - started_cpu,
    }
    return basis, report


def build_score_matrices(local, basis, holdout_x, plan, config):
    started_wall, started_cpu = perf_counter(), process_time()
    x = np.asarray(holdout_x, dtype=np.float64)
    validate_permutations(x, plan)
    if plan.shape[1] != config.summary_repeats:
        raise ValueError("holdout scoring repeats differ from the declared protocol")
    directions = basis / np.sqrt(score_metric(local.fisher, config))[:, None]
    tau, projections = tau_and_directional_derivatives(local.model, x, directions)
    baseline = np.column_stack((tau, projections))
    matrices = np.zeros((x.shape[1], basis.shape[1] + 1, basis.shape[1] + 1))
    for variable, _, changed in perturbed_covariates(x, plan):
        tau_changed, projections_changed = tau_and_directional_derivatives(local.model, changed, directions)
        difference = baseline - np.column_stack((tau_changed, projections_changed))
        matrices[variable] += difference.T @ difference / (2. * len(x) * plan.shape[1])
    if not np.isfinite(matrices).all():
        raise FloatingPointError("nonfinite holdout scoring matrices")
    return matrices, {
        "holdout_rows": len(x), "repeats": plan.shape[1], "variables": x.shape[1],
        "definition": "mean_H_and_repeats_delta_a_outer_delta_a_divided_by_2",
        "sample_weighting": "site_mean_only_target_pi_not_applied_locally",
        "wall_seconds": perf_counter() - started_wall,
        "cpu_seconds": process_time() - started_cpu,
    }


def quadratic_scores(theta, local_theta, fisher, basis, matrices, config):
    z = basis.T @ (np.sqrt(score_metric(fisher, config)) * (theta - local_theta))
    v = np.concatenate(([1.], z))
    scores = np.einsum("i,jik,k->j", v, matrices, v)
    allowance = 1e-10 * np.maximum(1., np.linalg.norm(matrices, axis=(1, 2)) * float(v @ v))
    if not np.isfinite(scores).all() or np.any(scores < -allowance):
        raise FloatingPointError("invalid PSD quadratic scoring result")
    return np.maximum(scores, 0.), z


def _average_ranks(scores):
    values, order = np.asarray(scores), np.argsort(scores, kind="stable")
    ranks = np.empty(len(values))
    for value in np.unique(values):
        positions = np.flatnonzero(values[order] == value)
        ranks[order[positions]] = positions.mean() + 1
    return ranks


def _top_inclusion(scores, k=2):
    values = np.asarray(scores)
    k = min(k, len(values))
    cutoff = np.sort(values)[-k]
    probabilities = (values > cutoff).astype(float)
    tied = values == cutoff
    probabilities[tied] = (k - probabilities.sum()) / tied.sum()
    return probabilities, k


def score_comparison(reference, estimate):
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    a, b = _average_ranks(reference), _average_ranks(estimate)
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    first, k = _top_inclusion(reference)
    second, _ = _top_inclusion(estimate)
    return {
        "per_variable_absolute_error": np.abs(reference - estimate).tolist(),
        "maximum_absolute_error": float(np.max(np.abs(reference - estimate))),
        "score_rmse": float(np.sqrt(np.square(reference - estimate).mean())),
        "spearman_average_tie_ranks": correlation,
        "top2_expected_overlap_fraction": float(first @ second / k),
        "tie_rule": "independent_uniform_boundary_tie_selections",
    }


def diagnostic_scores(local, theta_final, holdout_x, plan, config, basis=None, matrices=None):
    """Evaluator-only exact/full-gradient/optional low-rank comparison on same H."""
    started_wall, started_cpu = perf_counter(), process_time()
    x, theta = np.asarray(holdout_x, dtype=np.float64), np.asarray(theta_final, dtype=np.float64)
    validate_permutations(x, plan)
    target = model_from_vector(local.model_name, local.covariate_dim, theta)
    delta = theta - local.theta
    directions = delta[:, None]
    z = None
    if basis is not None:
        z = basis.T @ (np.sqrt(score_metric(local.fisher, config)) * delta)
        directions = np.column_stack((directions, basis / np.sqrt(score_metric(local.fisher, config))[:, None]))
    local_tau, projections = tau_and_directional_derivatives(local.model, x, directions)
    nonlinear = predict_tau(target, x)
    exact_tau_square_sum = float(np.square(nonlinear).sum())
    baselines = {"exact": nonlinear, "full_gradient": local_tau + projections[:, 0]}
    if basis is not None:
        baselines["low_rank"] = local_tau + projections[:, 1:] @ z
    scores = {key: np.zeros(x.shape[1]) for key in baselines}
    errors = {key: float(np.square(value - nonlinear).sum()) for key, value in baselines.items() if key != "exact"}
    observations = len(x)
    for variable, _, changed in perturbed_covariates(x, plan):
        tau_changed, projected = tau_and_directional_derivatives(local.model, changed, directions)
        exact_changed = predict_tau(target, changed)
        exact_tau_square_sum += float(np.square(exact_changed).sum())
        predictions = {"exact": exact_changed, "full_gradient": tau_changed + projected[:, 0]}
        if basis is not None:
            predictions["low_rank"] = tau_changed + projected[:, 1:] @ z
        for key, value in predictions.items():
            scores[key][variable] += float(np.square(baselines[key] - value).mean()) / (2. * plan.shape[1])
            if key != "exact":
                errors[key] += float(np.square(value - exact_changed).sum())
        observations += len(x)
    report = {
        "scope": "simulation_evaluator_only_same_H_permutations_site_means",
        "scores": {key: value.tolist() for key, value in scores.items()},
        "tau_rmse_over_original_and_all_interventions": {key: math.sqrt(value / observations) for key, value in errors.items()},
        "exact_tau_mean_square_over_original_and_all_interventions": exact_tau_square_sum / observations,
        "score_errors_against_nonlinear": {key: score_comparison(scores["exact"], value) for key, value in scores.items() if key != "exact"},
        "theta_euclidean_distance": float(np.linalg.norm(delta)),
        "theta_Fisher_distance": math.sqrt(float(np.dot(local.fisher, delta * delta))),
        "full_gradient_directional_derivatives": True,
        "parameter_square_matrix_allocated": False,
        "wall_seconds": perf_counter() - started_wall,
        "cpu_seconds": process_time() - started_cpu,
    }
    if matrices is not None:
        summary_scores, _ = quadratic_scores(theta, local.theta, local.fisher, basis, matrices, config)
        report["summary_vs_direct_low_rank_max_error"] = float(np.max(np.abs(summary_scores - scores["low_rank"])))
    return report
