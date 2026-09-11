"""Common-target adapter for the unchanged homogeneous CD R baselines.

T-only nuisance/local summaries and EBIC retain their historical numerical
definitions. The server target is explicitly changed to the common protocol:
sum_s pi_s/(2*n_Ts) (beta_s-beta)' precision_s (beta_s-beta).
V is unused; H provides only exact linear-model permutation scores. This is
a distinct result namespace, not a relabeling of historical R results.
"""
from dataclasses import asdict
import hashlib
import math
import time

import numpy as np

from effect_modifier_experiment.causal_comparison.rlearner import (
    CausalSite, LinearProbabilityEffectModel, RlearnerConfig, SiteSummary,
    _positive_definite, _validate_config, build_site_summary,
    crossfit_site_nuisances, fit_shared_effect, summary_cd_loss,
)
from .basis import BasisConfig, SharedBasis
from .dr_nuisance import array_hash
from .dr_sieve import _psd, holdout_score_blocks
from .interventions import permutation_hash, validate_permutations
from .types import SitePartitions
from .wire import communication_report, pack_payload, unpack_payload


def build_local_payload(site, config, seed, plan):
    if type(site) is not SitePartitions or not isinstance(config, RlearnerConfig):
        raise TypeError("observed SitePartitions and RlearnerConfig required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer nuisance seed required")
    _validate_config(config)
    basis = SharedBasis(BasisConfig("linear", site.train.n_features))
    validate_permutations(site.holdout.x, plan)
    if plan.shape[1] != 5:
        raise ValueError("five shared marginal holdout permutations required")
    local = CausalSite(site.train.x, site.train.a, site.train.y, site.site_id)
    wall, cpu = time.perf_counter(), time.process_time()
    detailed_stages = {}
    m, e, nuisance = crossfit_site_nuisances(local, config, seed, detailed_stages)
    nuisance_timing = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    wall, cpu = time.perf_counter(), time.process_time()
    summary, local_summary = build_site_summary(local, m, e, config)
    arrays = dict(beta=summary.beta, precision=summary.precision,
                  holdout_blocks=holdout_score_blocks(site.holdout.x, basis, plan))
    metadata = dict(protocol="r_common_target_v1", site_id=site.site_id, n_total=site.n_total,
                    n_train=len(site.train), n_validation=len(site.validation), n_holdout=len(site.holdout),
                    basis=basis.metadata(), config=asdict(config),
                    precision_rule="original_Z_transpose_Z_divided_by_RSS_over_n_train",
                    target_weight="n_total_over_sum_n_total",
                    permutation_sha256=permutation_hash(plan), perturbation_repeats=5,
                    effect_prediction_clipped=False)
    payload = pack_payload("r_common_target_v1", arrays, metadata)
    summary_timing = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    audit = dict(site_id=site.site_id, nuisance=nuisance, legacy_precision_summary=local_summary,
                 nuisance_component_stages=detailed_stages,
                 stages=dict(nuisance=nuisance_timing, local_summaries_and_serialization=summary_timing),
                 training_row_ids_sha256=array_hash(site.train.row_ids),
                 holdout_row_ids_sha256=array_hash(site.holdout.row_ids),
                 permutation_sha256=metadata["permutation_sha256"], payload_bytes=len(payload),
                 upload_messages=1, validation_used=False, holdout_fit_or_selection=False)
    return payload, audit


def adapted_summaries(payloads, config):
    """Summary-only normalization; the old solver divides by N_T once."""
    _validate_config(config)
    if not isinstance(payloads, (list, tuple)) or not payloads:
        raise ValueError("nonempty serialized R uploads required")
    expected = {"protocol", "site_id", "n_total", "n_train", "n_validation", "n_holdout", "basis", "config",
                "precision_rule", "target_weight", "permutation_sha256", "perturbation_repeats", "effect_prediction_clipped"}
    decoded, ids, common_basis = [], set(), None
    for payload in payloads:
        arrays, meta = unpack_payload(payload, "r_common_target_v1")
        if set(arrays) != {"beta", "precision", "holdout_blocks"} or set(meta) != expected:
            raise ValueError("unexpected R summary fields")
        if (meta["protocol"] != "r_common_target_v1" or meta["config"] != asdict(config)
                or meta["precision_rule"] != "original_Z_transpose_Z_divided_by_RSS_over_n_train"
                or meta["target_weight"] != "n_total_over_sum_n_total"
                or meta["perturbation_repeats"] != 5 or meta["effect_prediction_clipped"] is not False):
            raise ValueError("R common-target protocol mismatch")
        basis = SharedBasis(BasisConfig(**meta["basis"]["config"]))
        if basis.config.kind != "linear" or meta["basis"] != basis.metadata():
            raise ValueError("R baseline requires the common raw linear basis")
        if common_basis is not None and basis.config != common_basis.config:
            raise ValueError("R site covariate dimensions disagree")
        common_basis = basis
        site_id = meta["site_id"]
        if type(site_id) is not int or site_id < 0 or site_id in ids:
            raise ValueError("R site IDs must be nonnegative and unique")
        ids.add(site_id)
        counts = [meta[k] for k in ("n_train", "n_validation", "n_holdout", "n_total")]
        if any(type(n) is not int or n < 1 for n in counts) or sum(counts[:3]) != counts[3] or counts[0] <= basis.dimension:
            raise ValueError("invalid R counts or insufficient T rows for local OLS")
        if (arrays["beta"].shape != (basis.dimension,)
                or arrays["precision"].shape != (basis.dimension, basis.dimension)
                or arrays["holdout_blocks"].shape != (basis.config.covariate_dim, 1, 1)):
            raise ValueError("invalid R summary array dimensions")
        _positive_definite(arrays["precision"], config.condition_limit, "uploaded original R precision")
        for block in arrays["holdout_blocks"]:
            _psd(block, "R holdout score block")
        fingerprint = meta["permutation_sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            raise ValueError("invalid R permutation fingerprint")
        decoded.append((arrays, meta))
    decoded.sort(key=lambda item: item[1]["site_id"])
    n_train = sum(meta["n_train"] for _, meta in decoded)
    n_total = sum(meta["n_total"] for _, meta in decoded)
    weights = np.array([meta["n_total"]/n_total for _, meta in decoded])
    factors = np.array([n_train*meta["n_total"]/(n_total*meta["n_train"]) for _, meta in decoded])
    summaries = [SiteSummary(meta["site_id"], meta["n_train"], arrays["beta"], arrays["precision"]*factor)
                 for factor, (arrays, meta) in zip(factors, decoded)]
    blocks = sum(weight*arrays["holdout_blocks"] for weight, (arrays, _) in zip(weights, decoded))
    return summaries, blocks, dict(
        site_ids=[meta["site_id"] for _, meta in decoded], site_weights=weights.tolist(),
        precision_rescaling_factors=factors.tolist(), ebic_num_examples=n_train,
        target_num_examples=n_total, basis=common_basis.metadata(),
        counts=[dict(site_id=meta["site_id"], n_total=meta["n_total"], n_train=meta["n_train"],
                     n_validation=meta["n_validation"], n_holdout=meta["n_holdout"]) for _, meta in decoded])


def aggregate(payloads, penalty, config):
    wall, cpu = time.perf_counter(), time.process_time()
    summaries, blocks, target = adapted_summaries(payloads, config)
    model, fit = fit_shared_effect(summaries, penalty, config)
    scores = model.coefficients[1:]**2 * blocks[:, 0, 0]
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise RuntimeError("invalid R common holdout scores")
    report = dict(
        method="r_common_target_v1", penalty=penalty, config=asdict(config), fit=fit, target=target,
        converged=True, stop_reason=fit["stop_reason"],
        objective="sum_pi_s_over_2n_Ts_times_CD_precision_quadratic",
        weight_rule="old_precision*N_T*n_total_s/(N_total*n_Ts)_then_old_solver_divides_N_T",
        validation_used=False, holdout_fit_or_selection=False, refit_after_selection=False,
        effect_prediction_clipped=False,
        score_definition="sum_pi_s_mean_H_and_five_permutations_tau_difference_squared_over_two",
        communication=communication_report(payloads),
        stages=dict(server_validation_fit_selection_and_scores=dict(
            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)))
    return model, scores, report


def _kkt(beta, hessian, linear, lam, weights):
    gradient = hessian@beta-linear
    residual = np.maximum(np.abs(gradient)-lam*weights, 0.)
    active = beta != 0
    residual[active] = np.abs(gradient[active]+lam*weights[active]*np.sign(beta[active]))
    return float(np.max(residual))


def validate_server_result(payloads, model, scores, report, penalty, config):
    """Read-only selected-coefficient/KKT/CD/EBIC/score audit; never refit."""
    summaries, blocks, target = adapted_summaries(payloads, config)
    if penalty not in {"none", "adaptive_lasso"} or not isinstance(model, LinearProbabilityEffectModel):
        raise ValueError("invalid R baseline model or penalty")
    fixed = dict(method="r_common_target_v1", penalty=penalty, config=asdict(config), target=target,
                 converged=True, objective="sum_pi_s_over_2n_Ts_times_CD_precision_quadratic",
                 weight_rule="old_precision*N_T*n_total_s/(N_total*n_Ts)_then_old_solver_divides_N_T",
                 validation_used=False, holdout_fit_or_selection=False, refit_after_selection=False,
                 effect_prediction_clipped=False,
                 score_definition="sum_pi_s_mean_H_and_five_permutations_tau_difference_squared_over_two")
    if any(report[name] != value for name, value in fixed.items()):
        raise ValueError("R common-target identity or scope mismatch")
    fit = report["fit"]
    n, d = target["ebic_num_examples"], len(summaries[0].beta)
    hessian = sum(s.precision for s in summaries)/n
    linear = sum(s.precision@s.beta for s in summaries)/n
    beta = np.asarray(model.coefficients)
    initial = np.asarray(fit["unpenalized_shared_coefficients"], dtype=float)
    threshold = config.effect_tolerance*(1+float(np.max(np.abs(linear))))
    if (beta.shape != (d,) or initial.shape != (d,) or not np.isfinite(initial).all()
            or np.max(np.abs(hessian@initial-linear)) > threshold
            or not np.array_equal(beta, fit["shared_effect_coefficients"])
            or fit["effect_penalty"] != penalty or fit["converged"] is not True
            or fit["effect_intercept_penalized"] is not False or fit["num_examples"] != n
            or fit["num_sites"] != len(summaries) or fit["server_uses_individual_rows"] is not False
            or fit["additional_sample_size_multiplier"] is not False
            or fit["fit_fingerprint"] != hashlib.sha256(np.asarray(beta, dtype="<f8").tobytes()).hexdigest()):
        raise ValueError("invalid R shared-effect estimate or initializer")
    weights = np.r_[0., np.ones(d-1)] if penalty == "none" else np.r_[
        0., (np.abs(initial[1:])+config.adaptive_epsilon)**(-config.adaptive_power)]
    path = fit["lambda_fits"]
    if penalty == "none":
        lambdas = np.array([0.])
        selected = 0
        if fit["adaptive_weights"] is not None or fit["selection_rule"] != "no_selection" or not np.array_equal(beta, initial):
            raise ValueError("unpenalized R baseline changed its estimator")
    else:
        if not np.array_equal(weights, fit["adaptive_weights"]) or fit["selection_rule"] != "minimum_EBIC_ties_1e-12_largest_lambda":
            raise ValueError("invalid adaptive R weights or EBIC rule")
        intercept = np.zeros(d)
        intercept[0] = linear[0]/hessian[0, 0]
        lambda_max = float(np.max(np.abs((hessian@intercept-linear)[1:])/weights[1:]))
        lambdas = (np.geomspace(lambda_max, lambda_max*config.effect_lambda_min_ratio, config.effect_lambda_count)
                   if lambda_max > 0 else np.array([0.]))
        if lambdas[-1] != 0:
            lambdas = np.r_[lambdas, 0.]
        if not path or any(not np.isfinite(item["ebic"]) for item in path):
            raise ValueError("invalid R EBIC path")
        minimum = min(item["ebic"] for item in path)
        selected = next(i for i, item in enumerate(path) if item["ebic"] <= minimum+1e-12)
    if len(path) != len(lambdas) or not np.array_equal([item["lambda"] for item in path], lambdas):
        raise ValueError("R regularization path disagrees with declared grid")
    for item in path:
        if (item["converged"] is not True or item["stationarity_threshold"] != threshold
                or not np.isfinite(item["kkt_residual_inf"]) or not 0 <= item["kkt_residual_inf"] <= threshold
                or type(item["iterations"]) is not int or not 0 <= item["iterations"] <= config.effect_max_iter
                or item["stop_reason"] != ("closed_form" if item["lambda"] == 0 else "kkt_stationarity")):
            raise ValueError("invalid R convergence certificate")
    chosen = path[selected]
    residual = _kkt(beta, hessian, linear, float(lambdas[selected]), weights)
    if (type(fit["selected_index"]) is not int or fit["selected_index"] != selected
            or fit["selected_lambda"] != float(lambdas[selected]) or fit["selected_fit"] != chosen
            or report["stop_reason"] != chosen["stop_reason"] or fit["stop_reason"] != chosen["stop_reason"]
            or residual > threshold or not np.isclose(residual, chosen["kkt_residual_inf"], rtol=1e-9, atol=1e-12)):
        raise ValueError("R selected coefficient KKT or selection mismatch")
    if penalty == "adaptive_lasso":
        active = int(np.count_nonzero(beta[1:]))
        discrepancy = 2*n*summary_cd_loss(summaries, beta)
        ebic = discrepancy+(active+1)*math.log(n)+2*config.ebic_gamma*(math.lgamma(d)-math.lgamma(active+1)-math.lgamma(d-active))
        if (chosen["active_covariates"] != active or chosen["degrees_of_freedom"] != active+1
                or not np.isclose(chosen["cd_discrepancy"], discrepancy, rtol=1e-10, atol=1e-12)
                or not np.isclose(chosen["ebic"], ebic, rtol=1e-10, atol=1e-12)):
            raise ValueError("selected R EBIC does not use T count and common target discrepancy")
    expected_scores = beta[1:]**2*blocks[:, 0, 0]
    scores = np.asarray(scores)
    if scores.shape != (d-1,) or not np.allclose(scores, expected_scores, rtol=1e-10, atol=1e-12):
        raise ValueError("R holdout scores disagree with exact linear quadratic")
    if report["communication"] != communication_report(payloads):
        raise ValueError("R actual serialized communication mismatch")
    if set(report["stages"]) != {"server_validation_fit_selection_and_scores"}:
        raise ValueError("invalid R server timing stages")
    for stage in report["stages"].values():
        if any(not np.isfinite(stage[name]) or stage[name] < 0 for name in ("wall_seconds", "cpu_seconds")):
            raise ValueError("invalid R server timing")
