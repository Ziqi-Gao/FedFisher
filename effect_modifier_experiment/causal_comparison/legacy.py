"""Measured adapters for existing CPU methods; no experiment or CLI launch.

The dsLasso objective and existing local fitting/aggregation functions are
reused verbatim. Full selection/refit cost is observed, even when an old API
does not expose independent tuning and optimization stages.
"""

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from .. import dslassocov_benchmark as ds
from .. import one_shot_lasso_benchmark as one_shot
from ..one_shot_mlp_matched_fedavg import fit_matched_zero_penalty_site
from .evaluation import array_hash, model_arrays, model_hash
from .fedavg import _make_initial_model, _model_fingerprint
from .measurement import CommunicationLedger, MethodTimer


class DsCommunicationObserver:
    """Bounded-memory counts and ordered fingerprints of observed summaries."""

    def __init__(self):
        self._kinds = {}
        self._digest = hashlib.sha256(b"comparison-dslassocov-all-payloads-v1\0")

    def __call__(self, event):
        kind = event["kind"]
        if kind not in ("standardization", "loss_gradient", "loss_only"):
            raise ValueError("unknown dsLasso communication event")
        for field in ("num_sites", "vector_elements", "upload_bytes", "download_bytes"):
            if type(event[field]) is not int or event[field] <= 0:
                raise ValueError("invalid communication event %s" % field)
        payload_hash = event["payload_sha256"]
        if not isinstance(payload_hash, str) or len(payload_hash) != 64:
            raise ValueError("invalid communication payload fingerprint")
        int(payload_hash, 16)
        self._digest.update(json.dumps(event, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode() + b"\n")
        counts = self._kinds.setdefault(kind, dict(calls=0, upload_bytes=0,
                                                  download_bytes=0, site_messages=0))
        counts["calls"] += 1
        counts["upload_bytes"] += event["upload_bytes"]
        counts["download_bytes"] += event["download_bytes"]
        counts["site_messages"] += 2 * event["num_sites"]

    def to_dict(self):
        kinds = {kind: dict(counts) for kind, counts in self._kinds.items()}
        return dict(
            execution_model="single_process_simulation", actual_network_measured=False,
            counting_basis="observed numeric aggregation calls; not measured transport bytes",
            scope="entire fit: full/fold standardization, warm starts, all CV/final candidates, backtracking, validation and residual checks",
            payload_layout="float64 vectors/scalars; int64 site ID/count; means/scales and trial coefficients broadcast per site",
            excludes="transport framing, serialization overhead, preshared configuration, final model distribution and evaluation",
            accounted=bool(kinds), round_accounting_complete=True,
            identified_logical_rounds=sum(item["calls"] for item in kinds.values()),
            upload_bytes=sum(item["upload_bytes"] for item in kinds.values()),
            download_bytes=sum(item["download_bytes"] for item in kinds.values()),
            messages=sum(item["site_messages"] for item in kinds.values()),
            event_kinds=kinds, payload_fingerprint=self._digest.hexdigest(),
        )


def _validate_stationarity(report, label):
    if report.get("converged") is not True:
        raise RuntimeError("%s failed convergence" % label)
    threshold = float(report["stationarity_threshold"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise RuntimeError("%s has an invalid stationarity threshold" % label)
    for key in ("proximal_gradient_mapping_inf", "kkt_residual_inf"):
        residual = float(report[key])
        if not math.isfinite(residual) or residual < 0 or residual > threshold:
            raise RuntimeError("%s failed %s" % (label, key))


def _validated_ds_fit(fit):
    selected = fit["cv"]["selected_index"]
    if type(selected) is not int or not 0 <= selected < len(fit["cv"]["lambda_path"]):
        raise RuntimeError("invalid selected dsLasso candidate")
    folds = fit["cv"]["fold_reports"]
    if len(folds) != fit["cv"]["num_folds"]:
        raise RuntimeError("missing dsLasso CV fold evidence")
    for fold in folds:
        _validate_stationarity(fold["lambda_fits"][selected], "selected dsLasso CV fold")
    final = fit["optimizer"]["selected_lambda_fit"]
    _validate_stationarity(final, "selected dsLasso final fit")
    if final != fit["optimizer"]["final_lambda_fits"][selected]:
        raise RuntimeError("dsLasso final selection identity mismatch")
    return dict(selected_cv_folds_validated=len(folds), selected_final_fit_validated=True,
                selected_lambda=fit["cv"]["selected_lambda"],
                stationarity_scope="selected CV folds and selected final fit; original thresholds")


def _validate_model(model):
    arrays = model_arrays(model)
    if not arrays or not all(np.isfinite(values).all() for values in arrays.values()):
        raise RuntimeError("legacy method produced nonfinite model parameters")
    return model_hash(arrays)


def _fingerprint(record):
    return hashlib.sha256(json.dumps(ds.json_safe(record), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fit_legacy(raw_sites, site_datasets, method_config, seed, timer):
    """Return one validated legacy CPU model and complete, measured fit audit."""
    family = method_config["family"]
    if family not in ("dslassocov", "local_logistic_lasso", "one_shot_fedavg", "one_shot_group_lasso"):
        raise ValueError("unsupported legacy comparison family")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative integer below 2**63")
    timer = MethodTimer(family) if timer is None else timer
    options = dict(method_config["optimizer"])
    with timer.measure("train"):
        feature_dim = ds._validate_raw_sites(raw_sites)
        if feature_dim < 3 or feature_dim % 2 != 1:
            raise ValueError("legacy methods require [A, X, A*X] input")
        covariate_dim = (feature_dim - 1) // 2
        training_hash = ds.training_fingerprint(raw_sites)
        converted = ds.tensor_datasets_to_raw_sites(site_datasets, covariate_dim)
        if ds.training_fingerprint(converted) != training_hash:
            raise ValueError("raw sites and TensorDataset training sites disagree")
        initial_hash = None
        if family == "dslassocov":
            observer = DsCommunicationObserver()
            optimizer = ds.OptimizerConfig(
                max_iter=options["optimizer_max_iter"],
                tolerance=options["optimizer_tolerance"],
                initial_step=options["optimizer_initial_step"],
                backtrack_factor=options["optimizer_backtrack_factor"],
                min_step=options["optimizer_min_step"],
            )
            with ds.observe_communication(observer):
                fit = ds.federated_logistic_lasso_cv(
                    raw_sites, covariate_dim=covariate_dim,
                    num_folds=options["cv_folds"], lambda_count=options["lambda_count"],
                    lambda_min_ratio=options["lambda_min_ratio"],
                    cv_seed=options["cv_seed_offset"] + seed,
                    scale_floor=options["standardization_scale_floor"],
                    optimizer_config=optimizer,
                )
            convergence = _validated_ds_fit(fit)
            model = ds.FederatedLogisticLassoModel(fit["original_intercept"], fit["original_coefficients"])
            communication = observer.to_dict()
            if not communication["accounted"]:
                raise RuntimeError("dsLasso fit produced no communication observations")
            payload_hash = communication["payload_fingerprint"]
            site_audits = None
        else:
            sites = one_shot.tensor_datasets_to_sealed_sites(site_datasets, covariate_dim)
            if options.get("covariate_dim") != covariate_dim:
                raise ValueError("configured and supplied covariate dimensions differ")
            payloads, site_audits = [], []
            if family == "local_logistic_lasso":
                config = one_shot.LocalLogisticConfig(**options)
                common_state = None
            else:
                # Same CPU seed and model factory as multiround FedAvg. This
                # preserves the caller's RNG and does not touch CUDA state.
                config = one_shot.MlpGroupLassoConfig(**options)
                initial_model = _make_initial_model(method_config["model"], feature_dim, seed, None)
                initial_hash = _model_fingerprint(initial_model)
                common_state = {key: value.detach().clone() for key, value in initial_model.state_dict().items()}
            for site in sites:
                if family == "local_logistic_lasso":
                    payload, local_audit = one_shot.fit_local_logistic_lasso_site(site, config, seed)
                    for report in local_audit["selected_cv_reports"]:
                        _validate_stationarity(report, "selected local logistic CV fold")
                    if len(local_audit["selected_cv_reports"]) != config.cv_folds:
                        raise RuntimeError("missing local logistic CV fold evidence")
                    _validate_stationarity(local_audit["selected_final_report"], "selected local logistic final fit")
                elif family == "one_shot_group_lasso":
                    payload, local_audit = one_shot.fit_mlp_group_lasso_site(
                        site, method_config["model"], common_state, config, seed)
                else:
                    payload, local_audit = fit_matched_zero_penalty_site(
                        site, method_config["model"], common_state, config, seed)
                if not site.sealed:
                    raise RuntimeError("one-shot training site was not sealed after upload")
                payloads.append(payload)
                site_audits.append(local_audit)
            payload_hash = one_shot.payload_fingerprint(payloads)
            accesses_before = [site.data_accesses for site in sites]
            payload_metadata = []
            for payload, local_audit in zip(payloads, site_audits):
                vector = payload.coefficients if family == "local_logistic_lasso" else payload.model_vector
                size = int(vector.nbytes + 16)
                if local_audit["payload_bytes"] != size:
                    raise RuntimeError("one-shot payload byte evidence disagrees with actual payload")
                item = dict(site_idx=payload.site_idx, num_examples=payload.num_examples,
                            elements=int(vector.size), payload_bytes=size,
                            vector_sha256=array_hash(vector))
                if family == "local_logistic_lasso":
                    item["coefficients"] = vector.tolist()
                payload_metadata.append(item)
            convergence = dict(
                selected_cv_and_final_stationarity_validated=family == "local_logistic_lasso",
                convergence_claim="selected convex fits pass original residual criteria" if family == "local_logistic_lasso"
                else "none: completed prespecified local epochs and finite payload validation",
            )

    if family != "dslassocov":
        with timer.measure("aggregation"):
            if family == "local_logistic_lasso":
                coefficients, legacy_communication = one_shot.aggregate_linear_payloads(payloads, covariate_dim)
                model = one_shot.OneShotLinearLogisticModel(coefficients)
            else:
                model, legacy_communication = one_shot.aggregate_mlp_payloads(
                    payloads, method_config["model"], feature_dim, common_state)
            if accesses_before != [site.data_accesses for site in sites]:
                raise RuntimeError("one-shot aggregation accessed sealed training rows")
        ledger = CommunicationLedger()
        ledger.add("upload", sum(item["payload_bytes"] for item in site_audits),
                   messages=len(payloads), round_id="one-shot-fit", label="model_vector_site_id_count")
        if common_state is not None:
            initial_bytes = sum(value.numel() * value.element_size() for value in common_state.values())
            ledger.add("download", initial_bytes * len(payloads), messages=len(payloads),
                       round_id="one-shot-fit", label="shared_initial_model")
        communication = ledger.to_dict()
        communication.update(
            training_communication_rounds=1,
            pre_upload_training_communication_rounds=0,
            final_model_distribution_included=False,
            initialization_broadcast="common CPU MLP state" if common_state is not None else "none: independent local logistic initialization",
            legacy_api_report=legacy_communication,
            payload_fingerprint=payload_hash,
        )
        fit = dict(site_fits=site_audits, optimizer=asdict(config), payloads=payload_metadata,
                   sealed_sites=[dict(site_idx=site.site_idx, sealed=site.sealed,
                                      accesses_before_aggregation=before,
                                      accesses_after_aggregation=site.data_accesses,
                                      upload_count=1) for site, before in zip(sites, accesses_before)],
                   per_site_payload_values_persisted=family == "local_logistic_lasso")

    model_fingerprint = _validate_model(model)
    # Reject nonfinite tuning diagnostics too; do not let a finite final vector
    # conceal invalid candidate losses or convergence evidence.
    fit_hash = _fingerprint(dict(method_config=method_config, seed=seed,
                                 training=training_hash, payload=payload_hash,
                                 model=model_fingerprint, fit=fit))
    audit = dict(
        method=family, model_name=method_config.get("model"), seed=seed,
        config=method_config, device="cpu", completion_eligible=True,
        termination_reason="validated_selected_fits" if family in ("dslassocov", "local_logistic_lasso")
        else "completed_prespecified_local_epoch_budget",
        convergence=convergence,
        training_fingerprint=training_hash, payload_fingerprint=payload_hash,
        model_fingerprint=model_fingerprint, fit_fingerprint=fit_hash,
        initial_model_fingerprint=initial_hash,
        fit_metadata=fit, communication=communication,
        timing=timer.to_dict(),
        timing_contract=dict(
            training_includes_all_selection_and_refits=True,
            tuning_separately_measured=False,
            aggregation_separately_measured=family != "dslassocov",
            ds_aggregation_in_training=family == "dslassocov",
            note="legacy fitting APIs combine training and CV/holdout selection; no zero-cost tuning claim",
            observer_overhead_included=family == "dslassocov",
        ),
    )
    validate_fit_report(audit, method_config, model_arrays(model))
    return model, audit


def validate_fit_report(audit, method_config, arrays):
    """Validate persisted scientific acceptance, independent of timer values.

    Logistic payload coefficients are retained and their mean can be replayed.
    MLP payload summaries retain identities/hashes only: this validates their
    dimensions, fit/selection/sealing evidence and the aggregate model hash,
    without claiming to reconstruct unretained local vectors.
    """
    family = method_config["family"]
    if family not in ("dslassocov", "local_logistic_lasso", "one_shot_fedavg", "one_shot_group_lasso"):
        raise ValueError("unsupported legacy validation family")
    if audit.get("completion_eligible") is not True or audit.get("method") != family or audit.get("device") != "cpu":
        raise ValueError("legacy fit is not a completed CPU method identity")
    if _fingerprint(audit["config"]) != _fingerprint(method_config):
        raise ValueError("legacy fit configuration mismatch")
    seed = audit["seed"]
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("invalid legacy seed")
    if not arrays or any(not np.isfinite(np.asarray(values)).all() for values in arrays.values()):
        raise ValueError("nonfinite legacy model arrays")
    if model_hash(arrays) != audit["model_fingerprint"]:
        raise ValueError("legacy aggregate model fingerprint mismatch")
    fit, options = audit["fit_metadata"], method_config["optimizer"]
    communication = audit["communication"]
    if communication.get("actual_network_measured") is not False or communication.get("execution_model") != "single_process_simulation":
        raise ValueError("legacy communication must describe a simulation")
    if communication.get("payload_fingerprint") != audit["payload_fingerprint"]:
        raise ValueError("legacy payload identity mismatch")
    expected_fit_hash = _fingerprint(dict(method_config=method_config, seed=seed,
        training=audit["training_fingerprint"], payload=audit["payload_fingerprint"],
        model=audit["model_fingerprint"], fit=fit))
    if expected_fit_hash != audit["fit_fingerprint"]:
        raise ValueError("legacy fit fingerprint mismatch")
    if family == "dslassocov":
        convergence = _validated_ds_fit(fit)
        if convergence != audit["convergence"]:
            raise ValueError("dsLasso acceptance evidence mismatch")
        for report_key, option_key in (("max_iter", "optimizer_max_iter"), ("tolerance", "optimizer_tolerance"),
                ("initial_step", "optimizer_initial_step"), ("backtrack_factor", "optimizer_backtrack_factor"),
                ("min_step", "optimizer_min_step")):
            if fit["optimizer"][report_key] != options[option_key]:
                raise ValueError("dsLasso optimizer differs from manifest")
        cv = fit["cv"]
        if (cv["num_folds"] != options["cv_folds"] or cv["lambda_count"] != options["lambda_count"]
                or cv["lambda_min_ratio"] != options["lambda_min_ratio"]
                or cv["cv_seed"] != options["cv_seed_offset"] + seed or cv["uses_test_data"] is not False
                or len(cv["lambda_path"]) != options["lambda_count"]
                or cv["selected_lambda"] != cv["lambda_path"][cv["selected_index"]]):
            raise ValueError("dsLasso CV configuration or selection identity mismatch")
        losses = np.asarray(cv["fold_validation_losses"], dtype=np.float64)
        counts = np.asarray(cv["fold_validation_counts"], dtype=np.float64)
        if losses.shape != (options["cv_folds"], options["lambda_count"]) or counts.shape != (options["cv_folds"],) or np.any(counts <= 0):
            raise ValueError("invalid dsLasso CV loss/count dimensions")
        averaged = np.average(losses, axis=0, weights=counts)
        if not np.array_equal(averaged, cv["aggregated_validation_losses"]):
            raise ValueError("dsLasso weighted validation loss mismatch")
        selected = int(np.flatnonzero(np.isclose(averaged, np.min(averaged), rtol=0, atol=1e-12))[0])
        if selected != cv["selected_index"]:
            raise ValueError("dsLasso minimum-validation selection rule mismatch")
        means = np.asarray(fit["standardization"]["means"], dtype=np.float64)
        scales = np.asarray(fit["standardization"]["scales"], dtype=np.float64)
        if fit["standardization"]["report"]["scale_floor"] != options["standardization_scale_floor"] or np.any(scales <= 0):
            raise ValueError("invalid dsLasso standardization")
        intercept, coefficients = ds.backtransform_coefficients(
            np.asarray(fit["standardized_coefficients"], dtype=np.float64), means, scales)
        if intercept != fit["original_intercept"] or not np.array_equal(coefficients, fit["original_coefficients"]):
            raise ValueError("dsLasso coefficient backtransform mismatch")
        if not np.array_equal(np.asarray(arrays["coefficients"]), coefficients.astype(np.float32)) or not np.array_equal(
                np.asarray(arrays["intercept"]), np.asarray(intercept, dtype=np.float32)):
            raise ValueError("dsLasso selected fit does not match aggregate model")
        threshold = options["optimizer_tolerance"] * (1.0 + float(np.max(np.abs(fit["standardized_coefficients"]))))
        if fit["optimizer"]["selected_lambda_fit"]["stationarity_threshold"] != threshold:
            raise ValueError("dsLasso final stationarity threshold differs from manifest tolerance")
        kinds = communication["event_kinds"]
        if set(kinds) != {"standardization", "loss_gradient", "loss_only"}:
            raise ValueError("dsLasso communication lacks aggregation phases")
        if kinds["standardization"]["calls"] != options["cv_folds"] + 1:
            raise ValueError("dsLasso standardization communication count mismatch")
        for key, child_key in (("identified_logical_rounds", "calls"), ("upload_bytes", "upload_bytes"),
                               ("download_bytes", "download_bytes"), ("messages", "site_messages")):
            if communication[key] != sum(item[child_key] for item in kinds.values()):
                raise ValueError("dsLasso communication totals mismatch")
    else:
        if _fingerprint(fit["optimizer"]) != _fingerprint(options):
            raise ValueError("one-shot optimizer differs from manifest")
        payloads, local_fits, sealed = fit["payloads"], fit["site_fits"], fit["sealed_sites"]
        count = len(payloads)
        if count < 1 or len(local_fits) != count or len(sealed) != count:
            raise ValueError("one-shot site evidence is incomplete")
        if [item["site_idx"] for item in payloads] != list(range(count)):
            raise ValueError("one-shot site identities are not unique and contiguous")
        if fit["per_site_payload_values_persisted"] is not (family == "local_logistic_lasso"):
            raise ValueError("one-shot payload persistence scope is mislabeled")
        vector_elements = sum(np.asarray(value).size for value in arrays.values())
        scalar_bytes = 8 if family == "local_logistic_lasso" else 4
        for payload, local, seal in zip(payloads, local_fits, sealed):
            if not isinstance(payload["vector_sha256"], str) or len(payload["vector_sha256"]) != 64:
                raise ValueError("invalid one-shot vector fingerprint")
            int(payload["vector_sha256"], 16)
            if not (payload["site_idx"] == local["site_idx"] == seal["site_idx"]):
                raise ValueError("one-shot site evidence identity mismatch")
            if (type(payload["num_examples"]) is not int or payload["num_examples"] < 1
                    or payload["num_examples"] != local["num_examples"]
                    or payload["elements"] != vector_elements or local["payload_elements"] != vector_elements
                    or payload["payload_bytes"] != vector_elements * scalar_bytes + 16
                    or local["payload_bytes"] != payload["payload_bytes"]):
                raise ValueError("one-shot payload size/count mismatch")
            if (seal["sealed"] is not True or seal["upload_count"] != 1
                    or seal["accesses_before_aggregation"] != seal["accesses_after_aggregation"]
                    or seal["accesses_before_aggregation"] < 1
                    or local["pre_upload_training_communication_rounds"] != 0):
                raise ValueError("one-shot seal or single-upload boundary violated")
            if family == "local_logistic_lasso":
                if len(local["selected_cv_reports"]) != options["cv_folds"]:
                    raise ValueError("local logistic selected CV evidence missing")
                for report in local["selected_cv_reports"] + [local["selected_final_report"]]:
                    _validate_stationarity(report, "selected local logistic fit")
                if (type(local["selected_index"]) is not int or len(local["lambda_path"]) != options["lambda_count"]
                        or not 0 <= local["selected_index"] < options["lambda_count"]):
                    raise ValueError("local logistic selection dimensions mismatch")
                if local["selected_lambda"] != local["lambda_path"][local["selected_index"]]:
                    raise ValueError("local logistic selected lambda mismatch")
            elif family == "one_shot_fedavg":
                for key in ("local_epochs", "batch_size", "learning_rate", "momentum"):
                    if local[key] != options[key]:
                        raise ValueError("one-shot FedAvg local training configuration mismatch")
                if local["lambda"] != 0 or local["proximal_steps"] != 0:
                    raise ValueError("unpenalized one-shot FedAvg has a penalty")
            else:
                candidates = local["lambda_candidates"]
                if [candidate["ratio"] for candidate in candidates] != list(options["lambda_ratios"]):
                    raise ValueError("MLP LASSO candidate grid mismatch")
                best = min(range(len(candidates)), key=lambda index: candidates[index]["validation_loss"])
                threshold = candidates[best]["validation_loss"] + candidates[best]["validation_standard_error"]
                eligible = [index for index, candidate in enumerate(candidates)
                            if candidate["validation_loss"] <= threshold + 1e-12]
                selected = max(eligible, key=lambda index: candidates[index]["ratio"])
                if (local["selected_index"] != selected or local["selected_lambda"] != candidates[selected]["lambda"]
                        or local["selected_ratio"] != candidates[selected]["ratio"]):
                    raise ValueError("MLP LASSO one-standard-error selection mismatch")
                for candidate in candidates:
                    if candidate["lambda"] != candidate["ratio"] * local["local_gradient_scale"]:
                        raise ValueError("MLP LASSO lambda scale mismatch")
        expected_upload = sum(item["payload_bytes"] for item in payloads)
        expected_download = 0 if family == "local_logistic_lasso" else count * vector_elements * 4
        if (communication["upload_bytes"] != expected_upload or communication["download_bytes"] != expected_download
                or communication["training_communication_rounds"] != 1
                or communication["pre_upload_training_communication_rounds"] != 0
                or communication["legacy_api_report"]["post_upload_training_site_accesses"] != 0):
            raise ValueError("one-shot communication contract mismatch")
        if family == "local_logistic_lasso":
            restored = [one_shot.LinearSitePayload(item["site_idx"], item["num_examples"], item["coefficients"])
                        for item in payloads]
            for item, payload in zip(payloads, restored):
                if array_hash(payload.coefficients) != item["vector_sha256"]:
                    raise ValueError("local logistic payload coefficient hash mismatch")
            if one_shot.payload_fingerprint(restored) != audit["payload_fingerprint"]:
                raise ValueError("local logistic combined payload mismatch")
            weights = np.asarray([item.num_examples for item in restored], dtype=np.float64)
            weights /= weights.sum()
            coefficients = np.zeros(vector_elements, dtype=np.float64)
            for weight, payload in zip(weights, restored):
                coefficients += weight * payload.coefficients
            if (not np.array_equal(np.asarray(arrays["intercept"]), np.asarray(coefficients[0], dtype=np.float32))
                    or not np.array_equal(np.asarray(arrays["coefficients"]), coefficients[1:].astype(np.float32))):
                raise ValueError("local logistic weighted payload mean differs from aggregate model")
