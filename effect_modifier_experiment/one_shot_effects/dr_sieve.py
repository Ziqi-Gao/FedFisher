"""One-upload DR-Sieve: site moments, adaptive group LASSO, exact H scores.

All local moments are raw sums; server weights each site mean exactly once
by n_total / sum(n_total). Only serialized data-only summaries cross the
server API. Local Gram matrices may be singular. Ridge regularizes only the
declared adaptive-weight initializer, never the group-LASSO fit objective.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np

from .basis import BasisConfig, SharedBasis
from .dr_nuisance import DRNuisanceConfig, array_hash, fit_dr_pseudo_outcomes
from .interventions import permutation_hash, validate_permutations
from .types import SitePartitions
from .wire import canonical, communication_report, pack_payload, unpack_payload


@dataclass(frozen=True)
class DRSieveConfig:
    ridge_initial: float = 1e-6
    adaptive_epsilon: float = 1e-6
    adaptive_power: float = 0.5
    lambda_count: int = 30
    lambda_min_ratio: float = 1e-4
    validation_tie_tolerance: float = 1e-12
    max_iter: int = 20000
    tolerance: float = 1e-7
    solve_tolerance: float = 1e-9
    perturbation_repeats: int = 5

    def __post_init__(self):
        for name in ("ridge_initial", "adaptive_epsilon", "adaptive_power", "tolerance", "solve_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        if type(self.lambda_count) is not int or self.lambda_count < 2:
            raise ValueError("lambda_count must be an integer >= 2")
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not 0 < self.lambda_min_ratio < 1:
            raise ValueError("lambda_min_ratio must be in (0,1)")
        if not np.isfinite(self.validation_tie_tolerance) or self.validation_tie_tolerance < 0:
            raise ValueError("validation tie tolerance must be nonnegative and finite")
        if type(self.perturbation_repeats) is not int or self.perturbation_repeats != 5:
            raise ValueError("the common protocol requires exactly five perturbations")


@dataclass(frozen=True)
class DRSieveModel:
    beta: np.ndarray
    basis_config: BasisConfig

    def __post_init__(self):
        beta = np.array(self.beta, dtype=np.float64, copy=True)
        if beta.shape != (SharedBasis(self.basis_config).dimension,) or not np.isfinite(beta).all():
            raise ValueError("finite coefficients matching the shared basis required")
        beta.setflags(write=False)
        object.__setattr__(self, "beta", beta)

    def predict_tau(self, x):
        """Probability-scale working effect, deliberately without clipping."""
        return SharedBasis(self.basis_config).transform(x) @ self.beta


def raw_moments(design, psi):
    design, psi = np.asarray(design, dtype=np.float64), np.asarray(psi, dtype=np.float64)
    if (design.ndim != 2 or len(design) < 1 or psi.shape != (len(design),)
            or not np.isfinite(design).all() or not np.isfinite(psi).all()
            or not np.array_equal(design[:, 0], np.ones(len(design)))):
        raise ValueError("nonempty finite design with intercept and aligned pseudo-outcomes required")
    return design.T @ design, design.T @ psi, float(psi @ psi)


def holdout_score_blocks(x, basis, perturbations):
    """Mean delta-phi outer products / 2, restricted to the changed block."""
    validate_permutations(x, perturbations)
    if perturbations.shape[1] != 5 or len(x) < 1:
        raise ValueError("five marginal holdout permutations required")
    blocks = basis.blocks(x)
    result = np.empty((basis.config.covariate_dim, basis.group_size, basis.group_size))
    for j in range(result.shape[0]):
        delta = blocks[:, j, :][None, :, :] - blocks[perturbations[j], j, :]
        flat = delta.reshape(-1, basis.group_size)
        result[j] = flat.T @ flat / (2 * flat.shape[0])
    return result


def build_local_payload(site, basis_config, nuisance_config, config, seed, perturbations):
    """One complete upload, with detailed local audit retained outside the wire."""
    if not isinstance(config, DRSieveConfig) or not isinstance(nuisance_config, DRNuisanceConfig):
        raise TypeError("DR-Sieve and nuisance configuration objects required")
    if type(site) is not SitePartitions:
        raise TypeError("local fitting accepts observed SitePartitions only")
    basis = SharedBasis(basis_config)
    if type(site.site_id) is not int or site.site_id < 0 or type(site.n_total) is not int:
        raise ValueError("nonnegative site ID and integer n_total required")
    counts = [len(rows.x) for rows in (site.train, site.validation, site.holdout)]
    if min(counts) < 1 or sum(counts) != site.n_total:
        raise ValueError("n_total must equal disjoint nonempty T/V/H row counts")
    ids = [np.asarray(rows.row_ids) for rows in (site.train, site.validation, site.holdout)]
    if np.unique(np.concatenate(ids)).size != site.n_total:
        raise ValueError("T/V/H row IDs must be disjoint")
    if any(rows.x.shape[1] != basis_config.covariate_dim for rows in (site.train, site.validation, site.holdout)):
        raise ValueError("site raw dimensions disagree with shared basis")
    validate_permutations(site.holdout.x, perturbations)
    if perturbations.shape[1] != config.perturbation_repeats:
        raise ValueError("holdout permutation repeat count disagrees with protocol")
    wall, cpu = time.perf_counter(), time.process_time()
    train_psi, validation_psi, nuisance = fit_dr_pseudo_outcomes(
        site.train, site.validation, nuisance_config, seed)
    nuisance_timing = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    wall, cpu = time.perf_counter(), time.process_time()
    arrays = {}
    for name, rows, psi in (("train", site.train, train_psi), ("validation", site.validation, validation_psi)):
        gram, linear, constant = raw_moments(basis.transform(rows.x), psi)
        arrays[name+"_gram"], arrays[name+"_linear"] = gram, linear
        arrays[name+"_constant"] = np.array([constant], dtype=np.float64)
    arrays["holdout_blocks"] = holdout_score_blocks(site.holdout.x, basis, perturbations)
    metadata = dict(protocol="one_shot_dr_sieve_v1", site_id=site.site_id, n_total=site.n_total,
                    n_train=counts[0], n_validation=counts[1], n_holdout=counts[2],
                    basis=basis.metadata(), config=asdict(config), nuisance_config=asdict(nuisance_config),
                    summary_normalization="T_V_raw_sums_H_mean_outer_product_divided_by_two",
                    target_weight="n_total_over_sum_n_total",
                    permutation_sha256=permutation_hash(perturbations),
                    perturbation_repeats=config.perturbation_repeats,
                    effect_prediction_clipped=False)
    payload = pack_payload("dr_sieve", arrays, metadata)
    summary_timing = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    audit = dict(site_id=site.site_id, nuisance=nuisance, counts=dict(T=counts[0], V=counts[1], H=counts[2]),
                 stages=dict(nuisance=nuisance_timing, local_summaries_and_serialization=summary_timing),
                 training_gram_sha256=array_hash(arrays["train_gram"]),
                 training_linear_sha256=array_hash(arrays["train_linear"]),
                 holdout_row_ids_sha256=array_hash(site.holdout.row_ids),
                 permutation_sha256=metadata["permutation_sha256"],
                 payload_bytes=len(payload), upload_messages=1,
                 upload_contains_individual_rows=False, holdout_fit_or_selection=False)
    return payload, audit


def _psd(matrix, name):
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(name + " must be symmetric")
    scale = max(1., float(np.linalg.norm(matrix, ord=np.inf)))
    if np.linalg.eigvalsh(matrix)[0] < -1e-10 * scale:
        raise ValueError(name + " must be positive semidefinite")


def _decode_payloads(payloads, config):
    if not isinstance(payloads, (list, tuple)) or not payloads:
        raise ValueError("a nonempty list of serialized site uploads is required")
    decoded, site_ids, shared = [], set(), None
    expected_arrays = {"train_gram", "train_linear", "train_constant", "validation_gram",
                       "validation_linear", "validation_constant", "holdout_blocks"}
    expected_metadata = {"protocol", "site_id", "n_total", "n_train", "n_validation", "n_holdout",
                         "basis", "config", "nuisance_config", "summary_normalization", "target_weight",
                         "permutation_sha256", "perturbation_repeats", "effect_prediction_clipped"}
    for payload in payloads:
        arrays, metadata = unpack_payload(payload, "dr_sieve")
        if set(arrays) != expected_arrays or set(metadata) != expected_metadata:
            raise ValueError("unexpected DR-Sieve payload fields")
        if (metadata["protocol"] != "one_shot_dr_sieve_v1" or metadata["config"] != asdict(config)
                or metadata["summary_normalization"] != "T_V_raw_sums_H_mean_outer_product_divided_by_two"
                or metadata["target_weight"] != "n_total_over_sum_n_total"
                or metadata["effect_prediction_clipped"] is not False
                or metadata["perturbation_repeats"] != 5):
            raise ValueError("DR-Sieve payload protocol mismatch")
        basis = SharedBasis(BasisConfig(**metadata["basis"]["config"]))
        if metadata["basis"] != basis.metadata():
            raise ValueError("invalid fixed basis metadata")
        nuisance = DRNuisanceConfig(**metadata["nuisance_config"])
        shared_key = canonical(dict(basis=basis.metadata(), nuisance=asdict(nuisance)))
        if shared is not None and shared_key != shared:
            raise ValueError("sites disagree on common basis/nuisance configuration")
        shared = shared_key
        site_id = metadata["site_id"]
        if type(site_id) is not int or site_id < 0 or site_id in site_ids:
            raise ValueError("unique nonnegative site IDs required")
        site_ids.add(site_id)
        counts = [metadata[key] for key in ("n_train", "n_validation", "n_holdout", "n_total")]
        if any(type(n) is not int or n < 1 for n in counts) or sum(counts[:3]) != counts[3]:
            raise ValueError("invalid T/V/H counts or n_total")
        fingerprint = metadata["permutation_sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("invalid permutation fingerprint")
        for split, n in (("train", counts[0]), ("validation", counts[1])):
            gram, linear, constant = [arrays[split+suffix] for suffix in ("_gram", "_linear", "_constant")]
            if gram.shape != (basis.dimension, basis.dimension) or linear.shape != (basis.dimension,) or constant.shape != (1,):
                raise ValueError("invalid moment array dimensions")
            if not np.isclose(gram[0, 0], n, rtol=0, atol=1e-9) or constant[0] < 0:
                raise ValueError("moments must be raw sums with one intercept per row")
            augmented = np.block([[gram, linear[:, None]], [linear[None, :], constant.reshape(1, 1)]])
            _psd(augmented, split + " augmented moments")
        blocks = arrays["holdout_blocks"]
        if blocks.shape != (basis.config.covariate_dim, basis.group_size, basis.group_size):
            raise ValueError("invalid block holdout score dimensions")
        for block in blocks:
            _psd(block, "holdout score block")
        decoded.append((arrays, metadata))
    return sorted(decoded, key=lambda item: item[1]["site_id"]), basis


def aggregate_moments(payloads, config):
    """Public summary-only objective diagnostic; no local rows are accepted."""
    if not isinstance(config, DRSieveConfig):
        raise TypeError("DRSieveConfig required")
    decoded, basis = _decode_payloads(payloads, config)
    total = sum(meta["n_total"] for _, meta in decoded)
    weights = np.array([meta["n_total"] / total for _, meta in decoded])
    result = dict(basis=basis, site_ids=[meta["site_id"] for _, meta in decoded], weights=weights)
    for split in ("train", "validation"):
        for suffix in ("_gram", "_linear", "_constant"):
            key = split + suffix
            result[key] = sum(weight * arrays[key] / meta["n_"+split]
                              for weight, (arrays, meta) in zip(weights, decoded))
    result["holdout_blocks"] = sum(weight * arrays["holdout_blocks"]
                                    for weight, (arrays, _) in zip(weights, decoded))
    result["metadata"] = [metadata for _, metadata in decoded]
    return result


def group_penalty(beta, weights, group_size):
    return float(np.dot(weights * np.sqrt(group_size),
                        np.linalg.norm(beta[1:].reshape(-1, group_size), axis=1)))


def group_objective(beta, gram, linear, lambda_value, weights, group_size):
    return float(.5 * beta @ gram @ beta - linear @ beta
                 + lambda_value * group_penalty(beta, weights, group_size))


def group_kkt_residual(beta, gram, linear, lambda_value, weights, group_size):
    gradient = gram @ beta - linear
    blocks = beta[1:].reshape(-1, group_size)
    block_gradients = gradient[1:].reshape(-1, group_size)
    norms = np.linalg.norm(blocks, axis=1)
    penalties = lambda_value * weights * np.sqrt(group_size)
    residuals = np.maximum(np.linalg.norm(block_gradients, axis=1)-penalties, 0.)
    active = norms > 0
    residuals[active] = np.linalg.norm(block_gradients[active]
                                     + penalties[active, None]*blocks[active]/norms[active, None], axis=1)
    return max(float(abs(gradient[0])), float(residuals.max()))


def _prox(vector, thresholds, group_size):
    result = vector.copy()
    blocks = result[1:].reshape(-1, group_size)
    norms = np.linalg.norm(blocks, axis=1)
    factors = np.zeros_like(norms)
    nonzero = norms > 0
    factors[nonzero] = np.maximum(1-thresholds[nonzero]/norms[nonzero], 0.)
    blocks *= factors[:, None]
    return result


def solve_group_lasso(gram, linear, weights, group_size, lambda_value, config, initial=None, lipschitz=None):
    """Monotone restarted FISTA with explicit group KKT certification."""
    dimension = len(linear)
    if (gram.shape != (dimension, dimension) or weights.shape != ((dimension-1)//group_size,)
            or (dimension-1) % group_size or np.any(weights <= 0)
            or not np.isfinite(weights).all() or not np.isfinite(lambda_value) or lambda_value < 0):
        raise ValueError("invalid group quadratic objective")
    if lipschitz is None:
        _psd(gram, "group quadratic Gram")
        lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("positive finite quadratic Lipschitz constant required")
    beta = np.zeros(dimension) if initial is None else np.array(initial, dtype=np.float64, copy=True)
    if beta.shape != (dimension,) or not np.isfinite(beta).all():
        raise ValueError("invalid group solver initialization")
    threshold = config.tolerance * max(1., float(np.linalg.norm(linear, ord=np.inf)))
    thresholds = lambda_value * weights * np.sqrt(group_size) / lipschitz
    y, momentum = beta.copy(), 1.
    objective = group_objective(beta, gram, linear, lambda_value, weights, group_size)
    for iteration in range(config.max_iter+1):
        kkt = group_kkt_residual(beta, gram, linear, lambda_value, weights, group_size)
        if kkt <= threshold:
            mapping = lipschitz * (beta-_prox(beta-(gram@beta-linear)/lipschitz, thresholds, group_size))
            mapping_norm = max(float(abs(mapping[0])), float(np.linalg.norm(mapping[1:].reshape(-1, group_size), axis=1).max()))
            if mapping_norm > threshold * (1+1e-6) + 1e-12:
                raise RuntimeError("group LASSO KKT/proximal certificate disagreement")
            return beta, dict(converged=True, iterations=iteration, kkt_residual=kkt,
                              proximal_gradient_residual=mapping_norm, stationarity_threshold=threshold,
                              objective=objective, stop_reason="group_kkt_tolerance")
        if iteration == config.max_iter:
            break
        candidate = _prox(y-(gram@y-linear)/lipschitz, thresholds, group_size)
        value = group_objective(candidate, gram, linear, lambda_value, weights, group_size)
        if value > objective + 1e-14 * max(1., abs(objective)):
            y, momentum = beta.copy(), 1.
            candidate = _prox(y-(gram@y-linear)/lipschitz, thresholds, group_size)
            value = group_objective(candidate, gram, linear, lambda_value, weights, group_size)
        if not np.isfinite(candidate).all() or not np.isfinite(value):
            raise RuntimeError("nonfinite group LASSO iterate")
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        y = candidate + (momentum-1)/next_momentum * (candidate-beta)
        beta, momentum, objective = candidate, next_momentum, value
    raise RuntimeError("DR-Sieve group LASSO safety iteration limit reached without KKT convergence")


def validation_risk(beta, gram, linear, constant):
    """Site-weighted mean square error (no one-half factor)."""
    result = float(beta @ gram @ beta - 2*linear @ beta + np.asarray(constant).reshape(-1)[0])
    scale = max(1., float(abs(beta @ gram @ beta)), float(abs(2*linear @ beta)),
                float(abs(np.asarray(constant).reshape(-1)[0])))
    if not np.isfinite(result) or result < -1e-10 * scale:
        raise RuntimeError("invalid validation quadratic risk")
    return max(result, 0.)


def aggregate(payloads, config):
    """Return un-clipped model, exact selected-model H scores, and JSON report."""
    wall, cpu = time.perf_counter(), time.process_time()
    moments = aggregate_moments(payloads, config)
    basis, gram, linear = moments["basis"], moments["train_gram"], moments["train_linear"]
    ridge = gram.copy()
    ridge.flat[::len(ridge)+1] += config.ridge_initial
    ridge[0, 0] -= config.ridge_initial  # shared intercept remains unpenalized
    try:
        initial = np.linalg.solve(ridge, linear)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("declared adaptive initializer solve failed") from exc
    residual = float(np.linalg.norm(ridge@initial-linear, ord=np.inf))
    solve_threshold = config.solve_tolerance * max(1., float(np.linalg.norm(linear, ord=np.inf)))
    if not np.isfinite(initial).all() or not np.isfinite(residual) or residual > solve_threshold:
        raise RuntimeError("adaptive initializer failed matrix residual check")
    norms = np.linalg.norm(initial[1:].reshape(-1, basis.group_size), axis=1)
    weights = (norms + config.adaptive_epsilon)**(-config.adaptive_power)
    intercept = np.zeros_like(linear)
    intercept[0] = linear[0]/gram[0, 0]
    gradient = gram@intercept-linear
    lambda_max = float(np.max(np.linalg.norm(gradient[1:].reshape(-1, basis.group_size), axis=1)
                             / (weights*np.sqrt(basis.group_size))))
    lambdas = (np.geomspace(lambda_max, lambda_max*config.lambda_min_ratio, config.lambda_count)
               if lambda_max > 0 else np.zeros(config.lambda_count))
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    solutions, path, risks, current = [], [], [], intercept
    for value in lambdas:
        current, certificate = solve_group_lasso(
            gram, linear, weights, basis.group_size, float(value), config,
            initial=current, lipschitz=lipschitz)
        risk = validation_risk(current, moments["validation_gram"], moments["validation_linear"], moments["validation_constant"])
        solutions.append(current.copy())
        risks.append(risk)
        path.append(dict(certificate, lambda_value=float(value), validation_risk=risk,
                         nonzero_groups=int(np.count_nonzero(np.linalg.norm(current[1:].reshape(-1, basis.group_size), axis=1)))))
    selected = int(np.flatnonzero(np.array(risks) <= min(risks)+config.validation_tie_tolerance)[0])
    model = DRSieveModel(solutions[selected], basis.config)
    blocks = model.beta[1:].reshape(-1, basis.group_size)
    scores = np.einsum("pi,pij,pj->p", blocks, moments["holdout_blocks"], blocks)
    if not np.isfinite(scores).all() or np.any(scores < -1e-10):
        raise RuntimeError("invalid exact holdout quadratic importance")
    scores = np.maximum(scores, 0.)
    report = dict(
        method="one_shot_dr_sieve", config=asdict(config), basis=basis.metadata(),
        converged=True, stop_reason="all_lambda_group_kkt_then_V_summary_risk_selection",
        lambda_max=lambda_max, lambda_path=lambdas.tolist(), path=path,
        coefficient_path=[value.tolist() for value in solutions],
        selected_index=selected, selected_lambda=float(lambdas[selected]),
        selected_at_largest_lambda=selected == 0, selected_at_smallest_lambda=selected == len(lambdas)-1,
        validation_risks=risks, validation_risk_definition="sum_pi_s_mean_V_s_squared_error",
        selection_rule="minimum_V_summary_risk_absolute_ties_choose_larger_lambda",
        lambda_zero_rule="only_when_lambda_max_exactly_zero_repeat_zero_for_declared_path_length",
        initial_coefficients=initial.tolist(), initializer="fixed_slope_only_ridge_no_intercept_penalty",
        initial_solve_residual=residual, initial_solve_threshold=solve_threshold,
        adaptive_weights=weights.tolist(), group_penalty="lambda*sum_omega_j*sqrt_q_j*norm_beta_j",
        site_ids=moments["site_ids"], site_weights=moments["weights"].tolist(),
        weight_rule="n_total_over_sum_n_total_once_after_split_mean",
        counts=[dict(site_id=m["site_id"], n_total=m["n_total"], n_train=m["n_train"],
                     n_validation=m["n_validation"], n_holdout=m["n_holdout"]) for m in moments["metadata"]],
        score_definition="sum_pi_s_mean_H_and_five_permutations_tau_difference_squared_over_two",
        score_exact_for="selected_T_model_fixed_shared_basis_unclipped_tau_and_same_H_permutations",
        refit_after_selection=False, holdout_fit_or_selection=False, effect_prediction_clipped=False,
        communication=communication_report(payloads),
        stages=dict(server_validation_fit_selection_and_scores=dict(
            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)))
    return model, scores, report


def validate_server_result(payloads, model, scores, report, config):
    """Read-only completion audit: quadratic arithmetic, never refitting/inference."""
    moments = aggregate_moments(payloads, config)
    basis, gram, linear = moments["basis"], moments["train_gram"], moments["train_linear"]
    if (not isinstance(model, DRSieveModel) or model.basis_config != basis.config
            or report["method"] != "one_shot_dr_sieve" or report["config"] != asdict(config)
            or report["basis"] != basis.metadata() or report["converged"] is not True
            or report["holdout_fit_or_selection"] is not False
            or report["refit_after_selection"] is not False or report["effect_prediction_clipped"] is not False):
        raise ValueError("DR-Sieve result identity or fit scope mismatch")
    fixed_fields = dict(
        stop_reason="all_lambda_group_kkt_then_V_summary_risk_selection",
        validation_risk_definition="sum_pi_s_mean_V_s_squared_error",
        selection_rule="minimum_V_summary_risk_absolute_ties_choose_larger_lambda",
        lambda_zero_rule="only_when_lambda_max_exactly_zero_repeat_zero_for_declared_path_length",
        initializer="fixed_slope_only_ridge_no_intercept_penalty",
        group_penalty="lambda*sum_omega_j*sqrt_q_j*norm_beta_j",
        weight_rule="n_total_over_sum_n_total_once_after_split_mean",
        score_definition="sum_pi_s_mean_H_and_five_permutations_tau_difference_squared_over_two",
        score_exact_for="selected_T_model_fixed_shared_basis_unclipped_tau_and_same_H_permutations")
    if any(report[name] != value for name, value in fixed_fields.items()):
        raise ValueError("DR-Sieve fit or score definition mismatch")
    initial = np.asarray(report["initial_coefficients"], dtype=np.float64)
    if initial.shape != linear.shape or not np.isfinite(initial).all():
        raise ValueError("invalid adaptive initializer certificate")
    ridge = gram.copy()
    ridge.flat[::len(ridge)+1] += config.ridge_initial
    ridge[0, 0] -= config.ridge_initial
    residual = float(np.linalg.norm(ridge@initial-linear, ord=np.inf))
    threshold = config.solve_tolerance * max(1., float(np.linalg.norm(linear, ord=np.inf)))
    if (residual > threshold or not np.isclose(report["initial_solve_residual"], residual, rtol=1e-10, atol=1e-12)
            or report["initial_solve_threshold"] != threshold):
        raise ValueError("invalid adaptive initializer residual")
    weights = (np.linalg.norm(initial[1:].reshape(-1, basis.group_size), axis=1)
               + config.adaptive_epsilon)**(-config.adaptive_power)
    if not np.array_equal(weights, np.asarray(report["adaptive_weights"])):
        raise ValueError("adaptive weights disagree with declared initializer")
    intercept = np.zeros_like(linear)
    intercept[0] = linear[0]/gram[0, 0]
    gradient = gram@intercept-linear
    lambda_max = float(np.max(np.linalg.norm(gradient[1:].reshape(-1, basis.group_size), axis=1)
                             / (weights*np.sqrt(basis.group_size))))
    lambdas = (np.geomspace(lambda_max, lambda_max*config.lambda_min_ratio, config.lambda_count)
               if lambda_max > 0 else np.zeros(config.lambda_count))
    coefficients = np.asarray(report["coefficient_path"], dtype=np.float64)
    if (report["lambda_max"] != lambda_max or not np.array_equal(lambdas, np.asarray(report["lambda_path"]))
            or coefficients.shape != (config.lambda_count, basis.dimension)
            or not np.isfinite(coefficients).all() or len(report["path"]) != config.lambda_count):
        raise ValueError("invalid declared effect regularization path")
    risks = []
    kkt_threshold = config.tolerance * max(1., float(np.linalg.norm(linear, ord=np.inf)))
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    for beta, value, item in zip(coefficients, lambdas, report["path"]):
        kkt = group_kkt_residual(beta, gram, linear, float(value), weights, basis.group_size)
        risk = validation_risk(beta, moments["validation_gram"], moments["validation_linear"], moments["validation_constant"])
        objective = group_objective(beta, gram, linear, float(value), weights, basis.group_size)
        mapping = lipschitz*(beta-_prox(beta-(gram@beta-linear)/lipschitz,
                                       float(value)*weights*np.sqrt(basis.group_size)/lipschitz,
                                       basis.group_size))
        mapping_norm = max(float(abs(mapping[0])), float(np.linalg.norm(mapping[1:].reshape(-1, basis.group_size), axis=1).max()))
        if (item["converged"] is not True or item["stop_reason"] != "group_kkt_tolerance"
                or type(item["iterations"]) is not int or not 0 <= item["iterations"] <= config.max_iter
                or item["stationarity_threshold"] != kkt_threshold or kkt > kkt_threshold
                or not np.isclose(item["kkt_residual"], kkt, rtol=1e-9, atol=1e-12)
                or not np.isclose(item["objective"], objective, rtol=1e-10, atol=1e-12)
                or not np.isclose(item["validation_risk"], risk, rtol=1e-10, atol=1e-12)
                or item["lambda_value"] != float(value)
                or item["nonzero_groups"] != int(np.count_nonzero(np.linalg.norm(beta[1:].reshape(-1, basis.group_size), axis=1)))
                or not np.isfinite(item["proximal_gradient_residual"])
                or not np.isclose(item["proximal_gradient_residual"], mapping_norm, rtol=1e-9, atol=1e-12)
                or not 0 <= item["proximal_gradient_residual"] <= kkt_threshold*(1+1e-6)+1e-12):
            raise ValueError("group LASSO path certificate disagrees with summaries")
        risks.append(risk)
    selected = int(np.flatnonzero(np.array(risks) <= min(risks)+config.validation_tie_tolerance)[0])
    if (np.asarray(report["validation_risks"]).shape != (config.lambda_count,)
            or not np.allclose(report["validation_risks"], risks, rtol=1e-10, atol=1e-12)
            or type(report["selected_index"]) is not int or report["selected_index"] != selected
            or report["selected_lambda"] != float(lambdas[selected])
            or not np.array_equal(model.beta, coefficients[selected])
            or report["selected_at_largest_lambda"] is not (selected == 0)
            or report["selected_at_smallest_lambda"] is not (selected == len(lambdas)-1)):
        raise ValueError("validation risk selection or selected model mismatch")
    beta_blocks = model.beta[1:].reshape(-1, basis.group_size)
    expected_scores = np.maximum(np.einsum("pi,pij,pj->p", beta_blocks, moments["holdout_blocks"], beta_blocks), 0.)
    actual_scores = np.asarray(scores, dtype=np.float64)
    if actual_scores.shape != expected_scores.shape or not np.allclose(actual_scores, expected_scores, rtol=1e-10, atol=1e-12):
        raise ValueError("holdout scores disagree with exact un-clipped model quadratic")
    expected_counts = [dict(site_id=m["site_id"], n_total=m["n_total"], n_train=m["n_train"],
                            n_validation=m["n_validation"], n_holdout=m["n_holdout"])
                       for m in moments["metadata"]]
    if (report["site_ids"] != moments["site_ids"] or report["counts"] != expected_counts
            or not np.array_equal(report["site_weights"], moments["weights"])
            or report["communication"] != communication_report(payloads)):
        raise ValueError("site weighting or actual wire communication mismatch")
    if set(report["stages"]) != {"server_validation_fit_selection_and_scores"}:
        raise ValueError("invalid server timing stages")
    for stage in report["stages"].values():
        if any(not np.isfinite(stage[name]) or stage[name] < 0 for name in ("wall_seconds", "cpu_seconds")):
            raise ValueError("invalid server timing")
