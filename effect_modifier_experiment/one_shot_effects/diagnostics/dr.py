"""Evaluator-only replay of the frozen DR regularization paths.

No nuisance fitting, coefficient solving, or new model selection occurs here.
The scheduler composition validates the source artifacts and regenerated input
fingerprints before calling this module. Test truth is used only after the
original V-based selection and complete saved path have been validated.
"""
import time

import numpy as np

from ..dr_sieve import (
    DRSieveConfig, aggregate_moments, holdout_score_blocks,
    validate_server_result, validation_risk,
)
from ..evaluation import recovery
from ..interventions import permutation_hash, validate_permutations
from ..types import ExperimentData


def _ranking(scores):
    """Descending rank intervals preserve exact ties, including zero groups."""
    scores = np.asarray(scores, dtype=np.float64)
    minimum = np.array([1 + np.count_nonzero(scores > value) for value in scores])
    maximum = np.array([np.count_nonzero(scores >= value) for value in scores])
    return dict(
        rank_min=minimum.tolist(), rank_max=maximum.tolist(),
        average_rank=((minimum + maximum) / 2).tolist(),
        order_for_display=np.argsort(-scores, kind="stable").tolist(),
        indexing="zero_based_feature_indices; ranks_are_one_based",
        tie_rule="exact_score_ties_share_rank_interval; display_order_is_not_selection",
    )


def _test_weights(evaluation):
    if evaluation.truth is None or evaluation.site_ids is None:
        raise ValueError("DR replay requires SCMV2 evaluator truth and target site IDs")
    weights = np.zeros(len(evaluation.rows), dtype=np.float64)
    for site_id, weight in enumerate(evaluation.target_weights):
        indices = np.flatnonzero(evaluation.site_ids == site_id)
        if len(indices) < 2:
            raise ValueError("each target test site requires at least two rows")
        weights[indices] = weight / len(indices)
    return weights


def _path_summary(risks, nonzero, selected, tolerance):
    """Describe risk gaps without inventing an inferential flatness threshold."""
    minimum = float(np.min(risks))
    gaps = risks - minimum
    simpler = np.flatnonzero(nonzero < nonzero[selected])
    return dict(
        minimum_validation_risk=minimum,
        maximum_validation_risk=float(np.max(risks)),
        validation_risk_span=float(np.ptp(risks)),
        original_selected_risk_gap=float(gaps[selected]),
        absolute_tie_tolerance=tolerance,
        candidate_indices_in_original_V_tie_set=np.flatnonzero(gaps <= tolerance).tolist(),
        simpler_candidate_indices=simpler.tolist(),
        simpler_candidate_risk_gaps=gaps[simpler].tolist(),
        smallest_risk_gap_among_simpler_candidates=(
            float(np.min(gaps[simpler])) if len(simpler) else None),
        assessment="descriptive_risk_gaps_only; no_new_flatness_or_selection_threshold",
        risk_standard_error_available=False,
    )


def _candidate_summary(index, beta, value, risk, minimum, blocks, norms, scores, mask, selected):
    gap = float(risk - minimum)
    return dict(
        index=index, lambda_value=float(value), validation_risk=float(risk),
        validation_risk_gap_to_minimum=gap,
        validation_risk_relative_gap_to_minimum=gap / minimum if minimum > 0 else None,
        selected_by_original_V_rule=index == selected,
        nonzero_groups=int(np.count_nonzero(norms > 0)),
        nonzero_group_rule="Euclidean_group_coefficient_norm_strictly_greater_than_zero",
        intercept=float(beta[0]), coefficients=beta.tolist(),
        feature_groups=[dict(
            feature_index=j, feature_label="X" + str(j + 1), true_modifier=bool(mask[j]),
            coefficients=blocks[j].tolist(), coefficient_norm=float(norms[j]),
            nonzero=bool(norms[j] > 0), H_importance=float(scores[j]),
        ) for j in range(len(mask))],
        nonzero_true_modifier_groups=int(np.count_nonzero(norms[mask] > 0)),
        nonzero_other_groups=int(np.count_nonzero(norms[~mask] > 0)),
        H_scores=scores.tolist(), H_ranking=_ranking(scores), H_recovery=recovery(scores, mask),
    )


def _diagnose_basis(name, data, plans, source):
    original = source["config"]
    if (original["dataset"] != "scmv2_base" or original["seed"] != 0
            or original["recipe"] != "dr_" + name):
        raise ValueError("DR replay accepts only the frozen seed-zero SCMV2 base paths")
    configuration = DRSieveConfig(**original["dr_sieve"])
    if configuration.lambda_count != 30:
        raise ValueError("DR replay requires all thirty original lambda candidates")
    record = source["result"]["models"]["primary"]
    fit, model = record["fit"], source["models"]["primary"]
    payloads = source["payloads"]
    # This checks every coefficient/KKT/risk certificate and the unchanged V
    # selection by summary arithmetic. It never calls aggregate or a solver.
    validate_server_result(payloads, model, record["holdout"]["summary_squared_scores"],
                           fit, configuration)
    moments = aggregate_moments(payloads, configuration)
    basis = moments["basis"]
    expected_kind = "linear" if name == "linear" else "additive_cubic_bspline"
    if basis.config.kind != expected_kind:
        raise ValueError("source recipe disagrees with saved DR basis")
    if (moments["site_ids"] != [site.site_id for site in data.sites]
            or not np.array_equal(moments["weights"], data.evaluation.target_weights)):
        raise ValueError("saved DR summaries disagree with regenerated target sites")
    if len(plans) != len(data.sites):
        raise ValueError("one frozen H plan is required per site")
    rebuilt_blocks = np.zeros_like(moments["holdout_blocks"])
    for site, plan, metadata, weight in zip(
            data.sites, plans, moments["metadata"], moments["weights"]):
        validate_permutations(site.holdout.x, plan)
        if (permutation_hash(plan) != metadata["permutation_sha256"]
                or any(metadata["n_" + key] != len(getattr(site, key))
                       for key in ("train", "validation", "holdout"))
                or metadata["n_total"] != site.n_total):
            raise ValueError("regenerated DR partitions or H permutation fingerprints differ")
        rebuilt_blocks += weight * holdout_score_blocks(site.holdout.x, basis, plan)
    if not np.allclose(rebuilt_blocks, moments["holdout_blocks"], rtol=1e-10, atol=1e-12):
        raise ValueError("regenerated DR H blocks disagree with frozen uploaded summaries")

    coefficients = np.asarray(fit["coefficient_path"], dtype=np.float64)
    lambdas = np.asarray(fit["lambda_path"], dtype=np.float64)
    selected = fit["selected_index"]
    risks = np.array([validation_risk(beta, moments["validation_gram"],
                    moments["validation_linear"], moments["validation_constant"])
                      for beta in coefficients])
    selected_again = int(np.flatnonzero(
        risks <= np.min(risks) + configuration.validation_tie_tolerance)[0])
    if selected_again != selected or not np.array_equal(model.beta, coefficients[selected]):
        raise ValueError("original V-selected coefficient vector was not preserved")

    # All post-selection truth access is evaluator-only. Cache the basis once
    # rather than repeating spline transforms for thirty saved candidates.
    observation_weights = _test_weights(data.evaluation)
    mask = data.evaluation.truth.modifier_mask
    if (mask.shape != (basis.config.covariate_dim,)
            or not np.array_equal(mask, np.arange(basis.config.covariate_dim) < 2)):
        raise ValueError("evaluator modifier mask differs from shared feature coordinates")
    test_design = basis.transform(data.evaluation.rows.x)
    predicted_path = test_design @ coefficients.T
    if not np.isfinite(predicted_path).all():
        raise ValueError("nonfinite test prediction on saved coefficient path")
    truth = data.evaluation.truth.tau
    true_mean = float(observation_weights @ truth)
    true_variance = float(observation_weights @ ((truth - true_mean) ** 2))
    coefficient_blocks = coefficients[:, 1:].reshape(30, basis.config.covariate_dim, basis.group_size)
    norms = np.linalg.norm(coefficient_blocks, axis=2)
    nonzero = np.count_nonzero(norms > 0, axis=1)
    scores = np.einsum("kpi,pij,kpj->kp", coefficient_blocks,
                       moments["holdout_blocks"], coefficient_blocks)
    if not np.isfinite(scores).all() or np.any(scores < -1e-10):
        raise ValueError("invalid H importance on frozen coefficient path")
    scores = np.maximum(scores, 0.)
    selected_scores = np.asarray(record["holdout"]["summary_squared_scores"])
    if not np.allclose(scores[selected], selected_scores, rtol=1e-10, atol=1e-12):
        raise ValueError("selected path H scores differ from original selected result")

    minimum = float(np.min(risks))
    candidates = []
    for index, (beta, value, risk, prediction) in enumerate(zip(
            coefficients, lambdas, risks, predicted_path.T)):
        mean = float(observation_weights @ prediction)
        variance = float(observation_weights @ ((prediction - mean) ** 2))
        difference = prediction - truth
        item = _candidate_summary(index, beta, value, risk, minimum,
            coefficient_blocks[index], norms[index], scores[index], mask, selected)
        item["test"] = dict(
                cate_rmse=float(np.sqrt(observation_weights @ (difference ** 2))),
                prediction_mean=mean, true_mean=true_mean,
                mean_bias=float(observation_weights @ difference),
                estimated_heterogeneity_variance=variance,
                true_heterogeneity_variance=true_variance,
                out_of_bounds_fraction=float(observation_weights @ (np.abs(prediction) > 1)),
                prediction_clipped=False, diagnostic_only=True,
            )
        candidates.append(item)
    return dict(
        basis=basis.metadata(), candidate_count=len(candidates),
        original_recipe=original["recipe"], original_selected_index=selected,
        original_selected_lambda=fit["selected_lambda"],
        original_selection_rule=fit["selection_rule"], original_selection_preserved=True,
        path=candidates,
        validation_landscape=_path_summary(
            risks, nonzero, selected, configuration.validation_tie_tolerance),
        H_score_definition="sum_pi_s_mean_H_five_permutations_squared_tau_change_divided_by_two",
        H_summary_replay_max_absolute_error=float(
            np.max(np.abs(rebuilt_blocks - moments["holdout_blocks"]))),
        H_selected_score_replay_max_absolute_error=float(np.max(np.abs(scores[selected] - selected_scores))),
        test_weighting="original_target_pi_per_site_mean; no_empirical_site_frequency_reweighting",
        target_weights=moments["weights"].tolist(),
        true_modifier_mask=mask.tolist(),
        coefficient_interpretation="basis_specific_coefficients; group_norm_is_not_H_importance_or_significance",
    )


def diagnose_dr(config, data, plans, sources):
    """Return both frozen seed-zero DR paths; caller owns artifacts and timing."""
    if (config.get("dataset") != "scmv2_base" or config.get("seed") != 0
            or type(data) is not ExperimentData or set(sources) != {"linear", "spline"}):
        raise ValueError("DR replay requires both frozen SCMV2 base seed-zero paths")
    wall, cpu = time.perf_counter(), time.process_time()
    results = {name: _diagnose_basis(name, data, plans, sources[name])
               for name in ("linear", "spline")}
    return dict(
        diagnostic="dr_saved_regularization_path_replay_v1", basis_results=results,
        nuisance_refit=False, coefficient_refit=False, selection_changed=False,
        test_used_for_selection=False, holdout_used_for_selection=False,
        candidate_scope="all_thirty_original_lambda_candidates_for_each_saved_basis",
        interpretation=("nonzero_groups_are_not_significant_discoveries; no_FDR_control_or_"
                        "inference_claim; single_seed_descriptive_diagnostic"),
        timing=dict(wall_seconds=time.perf_counter() - wall,
                    cpu_seconds=time.process_time() - cpu),
    )


def _same(actual, expected, label):
    """Strict JSON structure, with tolerance only for independently replayed floats."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + " fields differ")
        for key in expected:
            _same(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + " list dimensions differ")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _same(left, right, label + "[" + str(index) + "]")
    elif isinstance(expected, float):
        if (type(actual) not in (int, float) or not np.isfinite(actual)
                or not np.isclose(actual, expected, rtol=1e-10, atol=1e-12)):
            raise ValueError(label + " numeric value differs")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(label + " value differs")


def _validate_test_observation(observation):
    numeric = {"cate_rmse", "prediction_mean", "true_mean", "mean_bias",
               "estimated_heterogeneity_variance", "true_heterogeneity_variance",
               "out_of_bounds_fraction"}
    if (not isinstance(observation, dict)
            or set(observation) != numeric | {"prediction_clipped", "diagnostic_only"}
            or observation["prediction_clipped"] is not False
            or observation["diagnostic_only"] is not True):
        raise ValueError("DR test diagnostic scope or fields differ")
    if any(type(observation[key]) not in (int, float) or not np.isfinite(observation[key])
           for key in numeric):
        raise ValueError("DR test observations must be finite")
    if (any(observation[key] < 0 for key in (
            "cate_rmse", "estimated_heterogeneity_variance", "true_heterogeneity_variance"))
            or not 0 <= observation["out_of_bounds_fraction"] <= 1
            or observation["cate_rmse"] + 1e-12 < abs(observation["mean_bias"])):
        raise ValueError("DR test observations violate moment bounds")
    _same(observation["prediction_mean"] - observation["true_mean"],
          float(observation["mean_bias"]), "DR test mean bias")


def validate_dr_diagnostic(report, config, sources):
    """Validate saved replay arithmetic without test inference or row transforms.

    Raw test moments remain measured observations; only their finite domain,
    moment identities, common truth moments, and selected-result agreement can
    be checked from these artifacts. This is not a second inference pass.
    """
    if (config.get("dataset") != "scmv2_base" or config.get("seed") != 0
            or set(sources) != {"linear", "spline"}):
        raise ValueError("invalid fixed DR replay validation scope")
    fixed = dict(
        diagnostic="dr_saved_regularization_path_replay_v1", nuisance_refit=False,
        coefficient_refit=False, selection_changed=False, test_used_for_selection=False,
        holdout_used_for_selection=False,
        candidate_scope="all_thirty_original_lambda_candidates_for_each_saved_basis",
        interpretation=("nonzero_groups_are_not_significant_discoveries; no_FDR_control_or_"
                        "inference_claim; single_seed_descriptive_diagnostic"),
    )
    if not isinstance(report, dict) or set(report) != set(fixed) | {"basis_results", "timing"}:
        raise ValueError("invalid DR replay result fields")
    for key, value in fixed.items():
        _same(report[key], value, key)
    if set(report["basis_results"]) != {"linear", "spline"}:
        raise ValueError("both original DR paths are required")
    if (set(report["timing"]) != {"wall_seconds", "cpu_seconds"}
            or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0
                   for v in report["timing"].values())):
        raise ValueError("invalid DR replay timing")
    common_truth = None
    for name in ("linear", "spline"):
        source, item = sources[name], report["basis_results"][name]
        original = source["config"]
        if (original["dataset"] != "scmv2_base" or original["seed"] != 0
                or original["recipe"] != "dr_" + name):
            raise ValueError("invalid original DR replay source scope")
        configuration = DRSieveConfig(**original["dr_sieve"])
        if configuration.lambda_count != 30:
            raise ValueError("all thirty original DR candidates are required")
        record = source["result"]["models"]["primary"]
        fit = record["fit"]
        validate_server_result(source["payloads"], source["models"]["primary"],
            record["holdout"]["summary_squared_scores"], fit, configuration)
        moments = aggregate_moments(source["payloads"], configuration)
        basis = moments["basis"]
        if basis.config.kind != ("linear" if name == "linear" else "additive_cubic_bspline"):
            raise ValueError("original DR basis differs from recipe")
        # The fixed SCMV2 base DGP has precisely X1/X2 as modifiers. This is
        # evaluator validation, never a fitting or tuning input.
        mask = np.arange(basis.config.covariate_dim) < 2
        coefficients = np.asarray(fit["coefficient_path"], dtype=np.float64)
        blocks = coefficients[:, 1:].reshape(30, len(mask), basis.group_size)
        norms = np.linalg.norm(blocks, axis=2)
        nonzero = np.count_nonzero(norms > 0, axis=1)
        risks = np.asarray([validation_risk(beta, moments["validation_gram"],
            moments["validation_linear"], moments["validation_constant"]) for beta in coefficients])
        selected = int(np.flatnonzero(risks <= np.min(risks) + configuration.validation_tie_tolerance)[0])
        if selected != fit["selected_index"]:
            raise ValueError("original DR V selection differs")
        scores = np.maximum(np.einsum("kpi,pij,kpj->kp", blocks, moments["holdout_blocks"], blocks), 0.)
        expected = dict(
            basis=basis.metadata(), candidate_count=30, original_recipe=original["recipe"],
            original_selected_index=selected, original_selected_lambda=fit["selected_lambda"],
            original_selection_rule=fit["selection_rule"], original_selection_preserved=True,
            validation_landscape=_path_summary(risks, nonzero, selected,
                                              configuration.validation_tie_tolerance),
            H_score_definition="sum_pi_s_mean_H_five_permutations_squared_tau_change_divided_by_two",
            H_selected_score_replay_max_absolute_error=float(np.max(np.abs(
                scores[selected] - np.asarray(record["holdout"]["summary_squared_scores"])))),
            test_weighting="original_target_pi_per_site_mean; no_empirical_site_frequency_reweighting",
            target_weights=moments["weights"].tolist(), true_modifier_mask=mask.tolist(),
            coefficient_interpretation="basis_specific_coefficients; group_norm_is_not_H_importance_or_significance",
        )
        if set(item) != set(expected) | {"path", "H_summary_replay_max_absolute_error"}:
            raise ValueError("DR path replay fields differ")
        for key, value in expected.items():
            _same(item[key], value, "DR " + name + "." + key)
        error = item["H_summary_replay_max_absolute_error"]
        block_scale = float(np.max(np.abs(moments["holdout_blocks"])))
        if (type(error) not in (int, float) or not np.isfinite(error)
                or not 0 <= error <= 1e-12 + 1e-10 * block_scale):
            raise ValueError("DR H summary regeneration check failed")
        if not isinstance(item["path"], list) or len(item["path"]) != 30:
            raise ValueError("DR path replay candidate count differs")
        for index, (candidate, beta, value, risk) in enumerate(zip(
                item["path"], coefficients, fit["lambda_path"], risks)):
            expected_candidate = _candidate_summary(index, beta, value, risk, float(np.min(risks)),
                blocks[index], norms[index], scores[index], mask, selected)
            if set(candidate) != set(expected_candidate) | {"test"}:
                raise ValueError("DR candidate replay fields differ")
            for key, expected_value in expected_candidate.items():
                _same(candidate[key], expected_value, "DR candidate " + str(index) + "." + key)
            observation = candidate["test"]
            _validate_test_observation(observation)
            truth_moments = [float(observation["true_mean"]), float(observation["true_heterogeneity_variance"])]
            if common_truth is None:
                common_truth = truth_moments
            _same(truth_moments, common_truth, "DR common independent test truth moments")
        selected_test = item["path"][selected]["test"]
        for new, old in (("cate_rmse", "cate_rmse"),
                         ("estimated_heterogeneity_variance", "estimated_heterogeneity"),
                         ("out_of_bounds_fraction", "effect_out_of_bounds_fraction")):
            _same(selected_test[new], float(record["test"][old]), "DR original selected test " + old)
