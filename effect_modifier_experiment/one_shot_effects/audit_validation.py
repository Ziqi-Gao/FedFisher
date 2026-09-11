"""Read-only local-fit evidence checks, without fitting or raw-row inference.

Observed row hashes bind reported inputs; they cannot prove that an optimizer
actually ran. Arrays use two explicitly named historical hash encodings so
the DR/R and Fisher records are never compared with composite row hashes.
"""
import hashlib
import math
import re

import numpy as np

from .wire import canonical, unpack_payload


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, message):
    _require(canonical(actual) == canonical(expected), message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _number(value, minimum=0.):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _array_hash(value, scheme):
    a = np.ascontiguousarray(value)
    header = (str((a.shape, a.dtype.str)).encode() if scheme == "nuisance_arrays"
              else str(a.shape).encode() + a.dtype.str.encode())
    return hashlib.sha256(header + a.tobytes()).hexdigest()


def observed_array_fingerprints(rows):
    """Runner-side observation hashes only; no treatment/outcome/truth use."""
    return {scheme: {"row_ids": _array_hash(rows.row_ids, scheme),
                     "raw_x": _array_hash(rows.x, scheme)}
            for scheme in ("nuisance_arrays", "fisher_arrays")}


def _stages(stages, required=None):
    _require(isinstance(stages, dict), "local timing stages must be a mapping")
    if required is not None:
        _same(sorted(stages), sorted(required), "local timing scope mismatch")
    for record in stages.values():
        _require(all(_number(record[name]) for name in ("wall_seconds", "cpu_seconds")),
                 "invalid local wall/CPU timing")
        if "calls" in record:
            _require(_integer(record["calls"], 1), "invalid local component call count")


def _fit_report(report, n, config, allow_constant=True, ids_hash=None):
    _require(_integer(n, 1) and report["training_rows"] == n and report["converged"] is True,
             "nuisance training count or convergence mismatch")
    if ids_hash is not None:
        _require(_hash(report["training_row_ids_sha256"]), "missing nuisance training ID hash")
        if ids_hash is not True:
            _same(report["training_row_ids_sha256"], ids_hash, "nuisance training IDs mismatch")
    kind = report["fit_kind"]
    if kind == "train_only_smoothed_bernoulli_constant":
        _require(allow_constant, "a constant propensity cannot replace required treatment overlap")
        prior = config["constant_pseudocount"]
        allowed = np.array([prior/(n+2*prior), (n+prior)/(n+2*prior)])
        probability = report["probability"]
        _require(_number(probability) and np.any(np.isclose(probability, allowed, rtol=0, atol=1e-14)),
                 "constant nuisance must use a declared train-only smoothed all-zero/all-one target")
        _same(report["pseudocount_per_class"], prior, "constant nuisance prior mismatch")
        _require(report["selected_lambda"] is None and report["optimizer_fits"] == 0,
                 "constant nuisance cannot claim a fitted or selected lambda")
        return False
    _require(kind == "site_local_logistic_lasso_nested_cv", "unsupported nuisance fit kind")
    folds, count = config["nuisance_cv_folds"], config["nuisance_lambda_count"]
    _require(n >= folds and report["num_folds"] == folds, "nuisance inner fold count mismatch")
    lambdas = np.geomspace(config["nuisance_lambda_max"],
                          config["nuisance_lambda_max"]*config["nuisance_lambda_min_ratio"], count)
    _same(report["lambda_path"], lambdas.tolist(), "nuisance fixed lambda grid mismatch")
    _same(report["lambda_grid_rule"], "fixed_absolute_grid_before_seeing_any_rows", "data-dependent nuisance grid")
    _same(report["selection_rule"], "minimum_row_weighted_logloss_ties_1e-12_largest_lambda", "nuisance CV rule mismatch")
    _same(report["standardization_scope"], "each_inner_training_fold_then_outer_training_refit_only", "nuisance standardization scope mismatch")
    _same(report["constant_pseudocount_per_class"], config["constant_pseudocount"], "inner constant prior mismatch")
    losses = np.asarray(report["validation_mean_logloss"], dtype=float)
    _require(losses.shape == (count,) and np.isfinite(losses).all() and np.all(losses >= 0),
             "invalid nuisance validation logloss")
    selected = int(np.flatnonzero(losses <= losses.min()+1e-12)[0])
    _require(type(report["selected_index"]) is int and report["selected_index"] == selected
             and report["selected_lambda"] == float(lambdas[selected]), "nuisance lambda choice mismatch")
    constants = report["constant_inner_folds"]
    _require(isinstance(constants, list) and all(_integer(f) and f < folds for f in constants)
             and constants == sorted(set(constants)), "invalid constant inner-fold list")
    fits = (folds-len(constants))*count+1
    _require(report["optimizer_fits"] == fits and _integer(report["optimizer_iterations"])
             and report["optimizer_iterations"] <= fits*config["nuisance_max_iter"]
             and _number(report["max_stationarity_ratio"])
             and report["max_stationarity_ratio"] <= 1., "nuisance optimizer safety/convergence certificate mismatch")
    return True


def _nuisance_stages(stages, model_count, cv_count):
    expected = {"nuisance_refit"} | ({"nuisance_cv"} if cv_count else set())
    _stages(stages, expected)
    _require(stages["nuisance_refit"]["calls"] == model_count, "nuisance refit timing count mismatch")
    if cv_count:
        _require(stages["nuisance_cv"]["calls"] == cv_count, "nuisance CV timing count mismatch")


def _dr(audit, arrays, meta, evidence, config, payload):
    nuisance, declared = audit["nuisance"], config["dr_nuisance"]
    _same(meta["nuisance_config"], declared, "DR upload nuisance config differs from reviewed recipe")
    _same(nuisance["config"], declared, "DR local nuisance config differs from reviewed recipe")
    _require(nuisance["seed"] == config["seed"] and nuisance["converged"] is True
             and nuisance["holdout_used"] is False, "DR nuisance seed, scope, or convergence mismatch")
    fixed = dict(validation_fit_scope="all_T_only",
        propensity_clipping="clip_predictions_to_declared_bounds_before_AIPW",
        failure_rule="empty_training_arm_or_nonconstant_target_smaller_than_inner_folds_fail",
        constant_outcome_rule="nonempty_train_only_smoothed_Bernoulli_with_declared_pseudocount",
        standardization_scope="each_inner_training_fold_then_outer_training_refit_only")
    for key, value in fixed.items():
        _same(nuisance[key], value, "DR nuisance policy mismatch: " + key)
    n, v = evidence["counts"]["train"], evidence["counts"]["validation"]
    hashes = evidence["array_fingerprints"]
    _same(nuisance["training_row_ids_sha256"], hashes["train"]["nuisance_arrays"]["row_ids"], "DR T ID binding mismatch")
    _same(nuisance["validation_row_ids_sha256"], hashes["validation"]["nuisance_arrays"]["row_ids"], "DR V ID binding mismatch")
    _same(audit["holdout_row_ids_sha256"], hashes["holdout"]["nuisance_arrays"]["row_ids"], "DR H ID binding mismatch")
    for key in ("fold_assignments_sha256", "training_psi_sha256", "validation_psi_sha256"):
        _require(_hash(nuisance[key]), "invalid DR crossfit/pseudo-outcome fingerprint")
    folds = nuisance["crossfit_folds"]
    f = declared["crossfit_folds"]
    _require(isinstance(folds, list) and len(folds) == f, "DR outer crossfit count mismatch")
    cv_count = 0
    total_arms = nuisance["treatment_arm_counts"]
    _require(isinstance(total_arms, list) and len(total_arms) == 2
             and all(_integer(value, 1) for value in total_arms) and sum(total_arms) == n,
             "DR complete T overlap/arm counts invalid")
    heldout_arm_totals = np.zeros(2, dtype=np.int64)
    for fold, item in enumerate(folds + [nuisance["validation_nuisance_refit"]]):
        held = n//f + int(fold < n % f) if fold < f else v
        train_n = n-held if fold < f else n
        _require(item["training_rows"] == train_n and item["prediction_rows"] == held
                 and item["converged"] is True, "DR outer training/prediction counts mismatch")
        if fold < f:
            _require(item["fold"] == fold and _hash(item["heldout_row_ids_sha256"]), "invalid DR outer fold identity")
        else:
            _same(item["training_row_ids_sha256"], nuisance["training_row_ids_sha256"], "V nuisance refit did not use full T")
        _require(_hash(item["training_row_ids_sha256"]), "invalid DR outer training ID fingerprint")
        arms = item["treatment_arm_counts"]
        _require(isinstance(arms, list) and len(arms) == 2 and all(_integer(value, 1) for value in arms)
                 and sum(arms) == train_n and all(a <= b for a, b in zip(arms, total_arms)),
                 "DR fold treatment overlap/arm counts invalid")
        if fold < f:
            heldout_arm_totals += np.array(total_arms)-arms
        else:
            _same(arms, total_arms, "DR full-T refit arm counts changed")
        _require(_integer(item["propensity_clip_count"]) and item["propensity_clip_count"] <= held
                 and _number(item["raw_propensity_min"]) and _number(item["raw_propensity_max"])
                 and item["raw_propensity_min"] <= item["raw_propensity_max"] <= 1., "DR propensity clipping evidence invalid")
        _same(sorted(item["models"]), ["mu0", "mu1", "propensity"], "DR requires three distinct nuisance targets")
        for name, target_n in (("propensity", train_n), ("mu0", arms[0]), ("mu1", arms[1])):
            cv_count += _fit_report(item["models"][name], target_n, declared,
                                    allow_constant=name != "propensity",
                                    ids_hash=item["training_row_ids_sha256"] if name == "propensity" else True)
    _same(heldout_arm_totals.tolist(), total_arms, "DR outer folds do not partition each treatment group once")
    _nuisance_stages(nuisance["stages"], 3*(f+1), cv_count)
    _same(audit["counts"], dict(T=n, V=v, H=evidence["counts"]["holdout"]), "DR local split count mismatch")
    for key, name in (("training_gram_sha256", "train_gram"), ("training_linear_sha256", "train_linear")):
        _same(audit[key], _array_hash(arrays[name], "nuisance_arrays"), "DR T moment fingerprint mismatch")
    _require(audit["payload_bytes"] == len(payload) and audit["upload_messages"] == 1
             and audit["upload_contains_individual_rows"] is False and audit["holdout_fit_or_selection"] is False,
             "DR local communication or holdout scope mismatch")
    _same(audit["permutation_sha256"], meta["permutation_sha256"], "DR local H plan differs from upload")
    _stages(audit["stages"], {"nuisance", "local_summaries_and_serialization"})


def _r(audit, arrays, meta, evidence, config, payload):
    declared, nuisance = config["r_baseline"], audit["nuisance"]
    _same(meta["config"], declared, "R upload configuration mismatch")
    n, f = evidence["counts"]["train"], declared["crossfit_folds"]
    _require(nuisance["crossfit_seed"] == config["seed"] and nuisance["test_rows_used"] == 0
             and nuisance["propensity_clip"] == declared["propensity_clip"]
             and _integer(nuisance["propensity_clipped_rows"])
             and nuisance["propensity_clipped_rows"] <= n, "R nuisance seed/clip/scope mismatch")
    _same(nuisance["fold_assignment_rule"], "balanced_shuffled_rows_independent_of_X_A_Y", "R fold rule mismatch")
    _require(_hash(nuisance["fold_assignment_sha256"]), "R fold fingerprint missing")
    folds = nuisance["outer_folds"]
    _require(isinstance(folds, list) and len(folds) == f, "R outer crossfit count mismatch")
    cv_count = 0
    for fold, item in enumerate(folds):
        held = n//f + int(fold < n % f)
        _require(item["fold"] == fold and item["held_out_rows"] == held and item["training_rows"] == n-held,
                 "R outer fold counts mismatch")
        for name in ("outcome", "propensity"):
            cv_count += _fit_report(item[name], n-held, declared)
    _nuisance_stages(audit["nuisance_component_stages"], 2*f, cv_count)
    fingerprints = evidence["array_fingerprints"]
    _same(audit["training_row_ids_sha256"], fingerprints["train"]["nuisance_arrays"]["row_ids"], "R T ID binding mismatch")
    _same(audit["holdout_row_ids_sha256"], fingerprints["holdout"]["nuisance_arrays"]["row_ids"], "R H ID binding mismatch")
    summary = audit["legacy_precision_summary"]
    d = config["dgp"]["covariate_dim"]+1
    raw = (np.array([meta["site_id"], n], dtype="<i8").tobytes()
           + np.asarray(arrays["beta"], dtype="<f8").tobytes()
           + np.asarray(arrays["precision"], dtype="<f8").tobytes())
    _require(summary["site_idx"] == meta["site_id"] and summary["num_examples"] == n
             and summary["effect_dimension"] == d and summary["payload_bytes"] == len(raw)
             and summary["payload_sha256"] == hashlib.sha256(raw).hexdigest(), "R local precision summary binding mismatch")
    _require(_number(summary["gram_condition_number"], 1.)
             and summary["gram_condition_number"] <= declared["condition_limit"]
             and _number(summary["residual_variance"])
             and summary["residual_variance"] > declared["residual_variance_floor"]
             and _number(summary["local_stationarity_inf"])
             and _number(summary["local_stationarity_threshold"])
             and 0 < summary["local_stationarity_threshold"]
             and summary["local_stationarity_inf"] <= summary["local_stationarity_threshold"], "R local OLS/variance convergence failure")
    _same(summary["variance_denominator"], "n_not_n_minus_p", "R local variance rule mismatch")
    _same(summary["precision_rule"], "Z_transpose_Z_divided_by_RSS_over_n", "R local precision rule mismatch")
    _require(audit["payload_bytes"] == len(payload) and audit["upload_messages"] == 1
             and audit["validation_used"] is False and audit["holdout_fit_or_selection"] is False,
             "R local upload or validation scope mismatch")
    _same(audit["permutation_sha256"], meta["permutation_sha256"], "R local H plan differs from upload")
    _stages(audit["stages"], {"nuisance", "local_summaries_and_serialization"})


def _em(audit, arrays, meta, evidence, config):
    declared, definition = config["fisher"], config["recipe_definition"]
    family, n = definition["family"], evidence["counts"]["train"]
    _same(meta["config"], declared, "outcome local configuration mismatch")
    fixed = dict(training_scope="T_only_fixed_epochs_no_V_or_H_selection_or_refit",
        model_name=definition["model"], covariate_dim=config["dgp"]["covariate_dim"],
        site_id=evidence["site_id"], seed=config["seed"], n_train=n,
        parameter_count=arrays["theta"].size, local_epochs=declared["local_epochs"],
        gradient_updates=declared["local_epochs"]*((n+declared["batch_size"]-1)//declared["batch_size"]),
        training_examples_seen=n*declared["local_epochs"],
        loader_seed=declared["loader_seed_offset"]+config["seed"]+100003*evidence["site_id"],
        optimizer="SGD", momentum=declared["momentum"],
        input_coordinates="shared_raw_[A,X,A_times_X]_no_site_standardization",
        precision="float64_after_common_float32_initialization", quantization="none",
        fisher_uses_observed_labels=False)
    for key, value in fixed.items():
        _same(audit[key], value, "outcome local fit evidence mismatch: " + key)
    hashes = evidence["array_fingerprints"]
    _same(audit["train_row_ids_sha256"], hashes["train"]["fisher_arrays"]["row_ids"], "outcome T ID binding mismatch")
    _same(audit["train_x_sha256"], hashes["train"]["fisher_arrays"]["raw_x"], "outcome T raw-X binding mismatch")
    _same(audit["theta_sha256"], _array_hash(arrays["theta"], "fisher_arrays"), "local parameter hash mismatch")
    _require(_hash(audit["initial_theta_sha256"]), "common initialization fingerprint missing")
    for key in ("training_wall_seconds", "training_cpu_seconds", "fisher_wall_seconds", "fisher_cpu_seconds",
                "full_local_wall_seconds", "full_local_cpu_seconds"):
        _require(_number(audit[key]), "invalid local outcome timing")
    if family == "fedavg":
        _require(audit["fisher_status"] == "not_computed" and audit["fisher_examples"] == 0
                 and audit["fisher_normalization"] is None, "FedAvg must not claim Fisher training")
        _same(audit["fisher_sha256"], _array_hash(np.zeros_like(arrays["theta"]), "fisher_arrays"), "uncomputed Fisher placeholder mismatch")
    else:
        _same(meta["local_fit"], audit, "uploaded local fit evidence differs from retained audit")
        _require(meta["seed"] == config["seed"] and meta["initial_theta_sha256"] == audit["initial_theta_sha256"]
                 and audit["fisher_status"] == "computed_mean_true_categorical_diag"
                 and audit["fisher_examples"] == n and audit["fisher_normalization"] == "divide_by_n_T_once",
                 "Fisher initialization, scope, or normalization mismatch")
        _same(audit["fisher_sha256"], _array_hash(arrays["fisher"], "fisher_arrays"), "local Fisher hash mismatch")
    if family == "compression":
        _same(meta["holdout_x_sha256"], hashes["holdout"]["fisher_arrays"]["raw_x"], "Fisher H raw-X binding mismatch")
        basis, summary = meta["basis"], meta["summary"]
        rows = min(n, declared["basis_max_rows"])
        random = declared["compression"] == "random_orthogonal"
        fields = dict(compression=declared["compression"], rank=declared["rank"],
            metric="identity" if declared["compression"] == "euclidean_gradient" else "local_mean_Fisher_plus_score_damping",
            score_damping=declared["score_damping"], aggregation_damping=0.,
            direction_seed=declared["direction_seed_offset"]+config["seed"],
            basis_sample_seed=declared["basis_seed_offset"]+config["seed"]+evidence["site_id"]*100003,
            basis_training_rows=rows, basis_max_rows=declared["basis_max_rows"],
            candidate_variables=config["dgp"]["covariate_dim"], perturbation_repeats_per_variable=declared["basis_repeats"],
            gradient_examples_per_pass=rows*(1+config["dgp"]["covariate_dim"]*declared["basis_repeats"]),
            gradient_passes=1 if random else 2, range_oversampling=declared["range_oversampling"],
            construction="fixed_seed_random_QR" if random else "two_pass_streamed_randomized_covariance_range_finder",
            matrix_free=True, parameter_square_matrix_allocated=False,
            error_scope="empirical_capped_training_gradient_collection_not_population_or_spectral_bound")
        for key, value in fields.items():
            _same(basis[key], value, "compression basis evidence mismatch: " + key)
        _require(_hash(basis["basis_training_row_ids_sha256"]), "compression T subset fingerprint missing")
        energy, captured = basis["observed_gradient_energy"], basis["observed_captured_energy"]
        _require(_number(energy) and _number(captured) and captured <= energy+1e-9*max(1., energy),
                 "invalid observed basis energy")
        expected_error = max(0., energy-captured)/energy if energy > 0 else 0.
        _require(_number(basis["observed_relative_projection_energy_error"])
                 and np.isclose(basis["observed_relative_projection_energy_error"], expected_error, rtol=1e-8, atol=1e-12),
                 "basis projection error evidence mismatch")
        if random:
            _require(basis["rayleigh_eigenvalues"] is None, "random basis cannot claim learned eigenvalues")
        else:
            eig = np.asarray(basis["rayleigh_eigenvalues"], dtype=float)
            _require(eig.shape == (declared["rank"],) and np.isfinite(eig).all(), "invalid basis Rayleigh evidence")
        _same(summary["holdout_rows"], evidence["counts"]["holdout"], "Fisher summary H count mismatch")
        _same(summary["repeats"], 5, "Fisher summary must use five perturbations")
        _same(summary["variables"], config["dgp"]["covariate_dim"], "Fisher summary variable count mismatch")
        _same(summary["definition"], "mean_H_and_repeats_delta_a_outer_delta_a_divided_by_2", "Fisher score definition mismatch")
        _same(summary["sample_weighting"], "site_mean_only_target_pi_not_applied_locally", "Fisher summary double weighting")
        _stages({"basis": basis, "summary": summary})
        _require(all(_number(meta[key]) for key in ("build_payload_wall_seconds", "build_payload_cpu_seconds")),
                 "invalid Fisher payload construction timing")


def validate_local_audits(result, payloads, config):
    """Validate saved local reports and common observed-input bindings only."""
    try:
        canonical(result["local_audits"])
        evidence, audits = result["data_fingerprints"]["sites"], result["local_audits"]
        _require(len(payloads) == len(audits) == len(evidence) == config["dgp"]["num_sites"],
                 "local audit/site count mismatch")
        family = config["recipe_definition"]["family"]
        initial_hashes = []
        for index, (site, audit, payload) in enumerate(zip(evidence, audits, payloads)):
            arrays, metadata = unpack_payload(payload)
            _require(audit["site_id"] == index == site["site_id"] == metadata["site_id"], "local audit/site identity mismatch")
            hashes = site["array_fingerprints"]
            _same(sorted(hashes), ["holdout", "train", "validation"], "observed split hash inventory mismatch")
            for split in hashes.values():
                _same(sorted(split), ["fisher_arrays", "nuisance_arrays"], "observed hash schemes mismatch")
                for values in split.values():
                    _same(sorted(values), ["raw_x", "row_ids"], "observed array hash inventory mismatch")
                    _require(all(_hash(value) for value in values.values()), "invalid observed array fingerprint")
            if family == "dr_sieve":
                _dr(audit, arrays, metadata, site, config, payload)
            elif family == "r_common":
                _r(audit, arrays, metadata, site, config, payload)
            else:
                _require(family in ("fedavg", "aggregation", "linearity", "compression"), "unknown local audit family")
                _em(audit, arrays, metadata, site, config)
                initial_hashes.append(audit["initial_theta_sha256"])
        if initial_hashes:
            _require(len(set(initial_hashes)) == 1, "sites did not share common initialization")
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("malformed local fit audit") from error
