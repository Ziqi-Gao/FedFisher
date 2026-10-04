"""Centralized pseudo-survival comparators, isolated from archived FL recipes.

Two response scopes answer different questions. ``site_local_pv`` centralizes
the original hospital-specific labels, whereas ``pooled_km_pv`` reconstructs
the Kaplan--Meier/IJ labels from all raw training records. Neither scope uses
test records, counterfactual truth, propensity weighting, or DR augmentation.
"""

from dataclasses import asdict
from hashlib import sha256
import math
import time

import numpy as np

from effect_modifier_experiment.survival.data import SurvivalSite
from effect_modifier_experiment.survival.dslasso import (
    DslassoConfig, SurvivalLinearModel, _path,
)
from effect_modifier_experiment.survival.links import survival_mean
from effect_modifier_experiment.survival.pseudo import pseudo_survival, site_pseudo_response
from effect_modifier_experiment.survival.recovery_method import (
    RECOVERY_SCHEMA_V2, OptimizationFailure, RecoveryPolicy, _RecoveryRun,
    _RejectedEvaluation, _validate_report,
)


FAMILIES = ("linear", "shallow", "deep")
PSEUDO_SCOPES = ("site_local_pv", "pooled_km_pv")
METHOD_SCHEMA = "survival_pooled_model_v1"


def array_hash(array):
    """Canonical float64 fingerprint shared by saved-model validators."""
    values = np.asarray(array, dtype="<f8", order="C")
    return sha256(str(values.shape).encode("ascii") + b"\0" + values.tobytes()).hexdigest()


def _training_inputs(sites, tau, seed, family, pseudo_scope):
    if (not sites or type(seed) is not int or seed < 0
            or family not in FAMILIES or pseudo_scope not in PSEUDO_SCOPES
            or not isinstance(tau, (int, float)) or isinstance(tau, bool)
            or not math.isfinite(tau) or tau <= 0):
        raise ValueError("invalid pooled training inputs")
    if (any(not isinstance(site, SurvivalSite) for site in sites)
            or [site.site_id for site in sites] != list(range(len(sites)))):
        raise ValueError("pooled training requires ordered distinct observed sites")
    rows = [site.rows for site in sites]
    if len({r.x.shape[1] for r in rows}) != 1:
        raise ValueError("inconsistent pooled covariate dimensions")
    identifiers = np.concatenate([r.row_ids for r in rows])
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("pooled rows must have globally unique identifiers")
    return rows


def _responses(rows, tau, pseudo_scope):
    """Construct labels exclusively from the supplied partition's records.

    The pooled labels are split back into their original groups only to retain
    dsLasso's floating-point summation order. All records reside centrally;
    this arithmetic grouping does not imply federated communication.
    """
    if pseudo_scope == "site_local_pv":
        return [site_pseudo_response(r, tau) for r in rows]
    if pseudo_scope != "pooled_km_pv":
        raise ValueError("unknown pseudo-observation scope")
    target = pseudo_survival(np.concatenate([r.time for r in rows]),
                            np.concatenate([r.event for r in rows]), tau)
    boundaries = np.cumsum([len(r) for r in rows])[:-1]
    return list(np.split(target, boundaries))


def _options(method_config):
    if not isinstance(method_config, dict):
        raise ValueError("pooled method configuration must be a dictionary")
    optimizer = method_config.get("optimizer", method_config)
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer configuration must be a dictionary")
    return dict(optimizer)


def _prepare(rows, tau, pseudo_scope, config):
    # Like the FL implementation, standardize A, X and AX on training records
    # only, retain a separate intercept, and never clip individual pseudo-values.
    raw = [(r.design, y) for r, y in zip(rows, _responses(rows, tau, pseudo_scope))]
    count = sum(len(x) for x, _ in raw)
    width = raw[0][0].shape[1]
    means = sum((x.sum(axis=0) for x, _ in raw), np.zeros(width)) / count
    squares = sum((np.square(x).sum(axis=0) for x, _ in raw), np.zeros(width)) / count
    scales = np.sqrt(np.maximum(squares - means ** 2, 0))
    scales[scales < config.scale_floor] = 1
    prepared = [(np.column_stack((np.ones(len(x)), (x - means) / scales)), y)
                for x, y in raw]
    mean_response = np.clip(sum(float(y.sum()) for _, y in raw) / count, .05, .95)
    initial = np.zeros(width + 1)
    initial[0] = np.log(-np.log(mean_response))
    return prepared, means, scales, initial


def _fit_linear(sites, rows, tau, seed, pseudo_scope, method_config):
    options = _options(method_config)
    max_step = options.pop("max_step", 64.0)
    options.setdefault("max_iter", 16000)
    config = DslassoConfig(**options)
    config.validate()
    policy_options = dict(method_config.get("recovery_policy", {}))
    policy_options.setdefault("max_step", max_step)
    policy = RecoveryPolicy(**policy_options)
    policy.validate(config)
    if any(len(r) < 2 * config.num_folds for r in rows):
        raise ValueError("two validation observations per site/fold are required")
    started = time.perf_counter()
    fractions = np.geomspace(1., config.lambda_min_ratio, config.num_lambdas)
    assignments = []
    for site in sites:
        rng = np.random.default_rng(np.random.SeedSequence([seed, site.site_id, 28461]))
        labels = np.empty(len(site.rows), dtype=int)
        labels[rng.permutation(len(labels))] = np.arange(len(labels)) % config.num_folds
        assignments.append(labels)
    context = dict(family="linear", pseudo_scope=pseudo_scope,
                   execution_scope="centralized_training")
    observer = _RecoveryRun(config, policy, context)
    observer.initialize(config, [len(r) for r in rows], fractions, assignments, sites)
    observer.begin_stage("cross_validation")
    squared_error = np.zeros(config.num_lambdas)
    validation_n = 0
    cv_paths = []
    try:
        for fold in range(config.num_folds):
            training = [r.subset(np.flatnonzero(a != fold)) for r, a in zip(rows, assignments)]
            validation = [r.subset(np.flatnonzero(a == fold)) for r, a in zip(rows, assignments)]
            prepared, means, scales, initial = _prepare(training, tau, pseudo_scope, config)
            observer.begin_path("cv", fold, prepared, means, scales, training)
            solutions, report = _path(prepared, initial, rows[0].x.shape[1], fractions,
                                      config, None, observer)
            report["fold"] = fold
            cv_paths.append(report)
            # Construct a fresh validation KM/IJ response. In particular, do
            # not subset full-training labels, which leak survival records.
            targets = _responses(validation, tau, pseudo_scope)
            for site_index, (r, target) in enumerate(zip(validation, targets)):
                design = np.column_stack((np.ones(len(r)), (r.design - means) / scales))
                observer.validation_partition(site_index, r, design, target)
                for index, beta in enumerate(solutions):
                    eta = design @ beta
                    observer.check_validation_predictor(eta)
                    residual = survival_mean(eta) - target
                    squared_error[index] += float(residual @ residual)
                validation_n += len(r)
            observer.completed_validation(fold, squared_error, validation_n)
        cv_loss = squared_error / validation_n
        selected = int(np.argmin(cv_loss))  # descending grid: exact ties prefer more L1
        observer.begin_stage("refit")
        prepared, means, scales, initial = _prepare(rows, tau, pseudo_scope, config)
        observer.begin_path("refit", None, prepared, means, scales, rows)
        solutions, refit = _path(prepared, initial, rows[0].x.shape[1],
                                 fractions[:selected + 1], config, None, observer)
    except _RejectedEvaluation as error:
        try:
            observer.fail(str(error))
        except OptimizationFailure as failure:
            failure.diagnostics = observer.finalize(failure.diagnostics, False)
            raise
    except OptimizationFailure as error:
        error.diagnostics = observer.finalize(error.diagnostics, False)
        raise
    beta = solutions[-1]
    coefficients = beta[1:] / scales
    model = SurvivalLinearModel(float(beta[0] - means @ coefficients), coefficients)
    arrays = dict(intercept=np.array(model.intercept), coefficients=coefficients.copy(),
                  standardized_coefficients=beta.copy(), means=means.copy(), scales=scales.copy())
    diagnostics = dict(schema=METHOD_SCHEMA, family="linear", pseudo_scope=pseudo_scope,
        seed=seed, sample_count=sum(len(r) for r in rows), input_dim=rows[0].design.shape[1],
        objective="half_mean_squared_error_plus_interaction_l1",
        link="log_minus_log_survival", execution_scope="centralized_training",
        communication_uploads=0, optimizer_config=asdict(config),
        stop_reason="first_order_stationarity_not_global_optimum",
        selected_lambda_index=selected, selected_lambda_fraction=float(fractions[selected]),
        selected_lambda_value=float(refit["lambda_max"] * fractions[selected]),
        cv_loss=cv_loss.tolist(), cv_paths=cv_paths,
        refit_lambda_max=float(refit["lambda_max"]), final_fit=refit["fits"][-1],
        training_pseudo_sha256=array_hash(np.concatenate([y for _, y in prepared])),
        model_sha256=array_hash(np.r_[model.intercept, coefficients]),
        total_fit_seconds=time.perf_counter() - started)
    diagnostics = observer.finalize(diagnostics, True)
    # Observer accounting concerns arithmetic groups, not network operations.
    diagnostics["central_arithmetic_accounting"] = diagnostics.pop("communication")
    return model, diagnostics, arrays


def _fit_neural(rows, tau, seed, family, pseudo_scope, method_config, initial_eta):
    import torch
    from torch.nn.utils import parameters_to_vector
    from effect_modifier_experiment.survival.fisher import (
        FisherConfig, FisherFit, _load_vector, _mean, _model, public_initial_vector,
    )

    options = _options(method_config)
    options.setdefault("architecture", family)
    config = FisherConfig(**options)
    config.validate()
    if config.architecture != family:
        raise ValueError("pooled neural family and architecture disagree")
    if not math.isfinite(initial_eta):
        raise ValueError("initial_eta must be a finite public design constant")
    started = time.perf_counter()
    design = np.concatenate([r.design for r in rows])
    response = np.concatenate(_responses(rows, tau, pseudo_scope))
    initial = public_initial_vector(design.shape[1], config, seed, initial_eta)
    model = _model(design.shape[1], config)
    _load_vector(model, initial)
    x, target = torch.from_numpy(design.copy()), torch.from_numpy(response.copy())
    # The centralized comparator starts from exactly the FL public vector.
    # Its own shuffle stream is the first-site stream, declared independently
    # of responses/truth. A one-site fixture therefore reproduces FL SGD.
    training_seed = seed + 1009
    rng = torch.Generator(device="cpu").manual_seed(training_seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                momentum=config.momentum)
    updates = 0
    with torch.no_grad():
        initial_loss = float((_mean(model(x).squeeze(-1)) - target).square().mean()
                             / (2 * config.working_variance))
    for _ in range(config.epochs):
        permutation = torch.randperm(len(x), generator=rng)
        for start in range(0, len(x), config.batch_size):
            indices = permutation[start:start + config.batch_size]
            optimizer.zero_grad()
            loss = (_mean(model(x[indices]).squeeze(-1)) - target[indices]).square().mean()
            loss = loss / (2 * config.working_variance)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite pooled neural objective")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite pooled neural gradient")
            optimizer.step()
            updates += 1
    vector = parameters_to_vector(model.parameters()).detach().numpy().copy()
    if not np.isfinite(vector).all():
        raise FloatingPointError("nonfinite pooled neural parameters")
    with torch.no_grad():
        final_loss = float((_mean(model(x).squeeze(-1)) - target).square().mean()
                           / (2 * config.working_variance))
    if not math.isfinite(final_loss):
        raise FloatingPointError("nonfinite final pooled objective")
    diagnostics = dict(schema=METHOD_SCHEMA, family=family, pseudo_scope=pseudo_scope,
        seed=seed, training_seed=training_seed, sample_count=len(response), input_dim=design.shape[1],
        site_sizes=[len(r) for r in rows], optimizer_config=asdict(config),
        objective="half_mean_squared_error_divided_by_working_variance",
        link="log_minus_log_survival", execution_scope="centralized_training",
        communication_uploads=0, fisher_computed=False,
        epochs=config.epochs, optimizer_updates=updates,
        processed_rows=config.epochs * len(response), stop_reason="fixed_epoch_budget",
        initial_loss=initial_loss, final_loss=final_loss,
        initial_eta=float(initial_eta), initial_sha256=sha256(initial.tobytes()).hexdigest(),
        training_pseudo_sha256=array_hash(response), model_sha256=array_hash(vector),
        total_fit_seconds=time.perf_counter() - started)
    # FisherFit is reused only as a scalar-output prediction wrapper. No
    # curvature, upload, or server optimization is performed by this trainer.
    fitted = FisherFit(vector, config, design.shape[1], diagnostics, ())
    return fitted, diagnostics, dict(vector=vector, initial_vector=initial)


def fit_pooled(sites, tau, seed, family, pseudo_scope, method_config, initial_eta):
    """Fit a centralized comparator without accepting evaluator records/truth."""
    rows = _training_inputs(sites, tau, seed, family, pseudo_scope)
    if family == "linear":
        return _fit_linear(sites, rows, tau, seed, pseudo_scope, method_config)
    return _fit_neural(rows, tau, seed, family, pseudo_scope, method_config, initial_eta)


def validate_fit(diagnostics, arrays, *, family, pseudo_scope, method_config,
                 seed, site_sizes, initial_eta):
    """Validate saved optimizer evidence and parameters, without inference."""
    if (diagnostics.get("schema") != METHOD_SCHEMA or diagnostics.get("family") != family
            or diagnostics.get("pseudo_scope") != pseudo_scope
            or diagnostics.get("seed") != seed
            or diagnostics.get("execution_scope") != "centralized_training"
            or diagnostics.get("communication_uploads") != 0
            or diagnostics.get("sample_count") != sum(site_sizes)
            or diagnostics.get("site_sizes") != list(site_sizes)
            or diagnostics.get("link") != "log_minus_log_survival"):
        raise ValueError("pooled fit identity mismatch")
    if any(not np.isfinite(np.asarray(value)).all() for value in arrays.values()):
        raise ValueError("nonfinite pooled model arrays")
    if family == "linear":
        required = {"intercept", "coefficients", "standardized_coefficients", "means", "scales"}
        if set(arrays) != required:
            raise ValueError("invalid pooled linear array fields")
        options = _options(method_config)
        max_step = options.pop("max_step", 64.0)
        options.setdefault("max_iter", 16000)
        config = DslassoConfig(**options)
        config.validate()
        policy_options = dict(method_config.get("recovery_policy", {}))
        policy_options.setdefault("max_step", max_step)
        policy = RecoveryPolicy(**policy_options)
        policy.validate(config)
        if (diagnostics.get("optimizer_config") != asdict(config)
                or diagnostics.get("config") != asdict(config)
                or diagnostics.get("recovery_policy") != asdict(policy)
                or diagnostics.get("recovery_schema") != RECOVERY_SCHEMA_V2
                or diagnostics.get("outcome") != "stationary"
                or diagnostics.get("stop_reason") != "first_order_stationarity_not_global_optimum"):
            raise ValueError("invalid pooled linear optimizer recipe")
        fractions = np.geomspace(1., config.lambda_min_ratio, config.num_lambdas)
        losses = np.asarray(diagnostics["cv_loss"], dtype=float)
        selected = diagnostics["selected_lambda_index"]
        if (losses.shape != (config.num_lambdas,) or not np.isfinite(losses).all()
                or np.any(losses < 0) or type(selected) is not int
                or selected != int(np.argmin(losses))
                or diagnostics["lambda_fractions"] != fractions.tolist()
                or diagnostics["selected_lambda_fraction"] != float(fractions[selected])
                or diagnostics["selected_lambda_value"] != float(
                    diagnostics["refit_lambda_max"] * fractions[selected])):
            raise ValueError("invalid pooled linear CV selection")
        paths = diagnostics.get("paths", [])
        if len(paths) != config.num_folds + 1:
            raise ValueError("missing pooled linear path evidence")
        if (diagnostics.get("context") != dict(family="linear", pseudo_scope=pseudo_scope,
                                               execution_scope="centralized_training")
                or diagnostics.get("objective") != "half_mean_squared_error_plus_interaction_l1"
                or diagnostics.get("design_width") != diagnostics["input_dim"]
                or type(diagnostics["input_dim"]) is not int
                or diagnostics["input_dim"] < 5 or diagnostics["input_dim"] % 2 != 1):
            raise ValueError("invalid pooled linear context/design")
        progress = diagnostics.get("validation_progress", [])
        if (len(progress) != config.num_folds or progress[-1]["cumulative_n"] != sum(site_sizes)
                or not np.array_equal(losses, np.asarray(progress[-1]["cumulative_squared_error"])
                                      / progress[-1]["cumulative_n"])
                or len(diagnostics.get("cv_paths", [])) != config.num_folds):
            raise ValueError("pooled linear CV evidence is incomplete")
        for index, path in enumerate(paths):
            is_refit = index == config.num_folds
            if (path["stage"] != ("refit" if is_refit else "cv")
                    or path["fold"] != (None if is_refit else index)
                    or len(path["fits"]) != (selected + 1 if is_refit else config.num_lambdas)):
                raise ValueError("pooled linear path identity mismatch")
            if not np.isfinite(path["lambda_max"]) or path["lambda_max"] < 0:
                raise ValueError("invalid pooled lambda maximum")
            if not is_refit and diagnostics["cv_paths"][index] != dict(
                    lambda_max=path["lambda_max"], unpenalized_start=path["unpenalized_start"],
                    fits=path["fits"], fold=index):
                raise ValueError("pooled linear CV path copies disagree")
            for fit_index, report in enumerate([path["unpenalized_start"]] + path["fits"]):
                unpenalized = fit_index == 0
                lambda_index = fit_index - 1
                penalty_start = (diagnostics["input_dim"] + 3) // 2
                expected_context = dict(diagnostics["context"], stage=path["stage"],
                    fold=path["fold"], phase=("unpenalized_start" if unpenalized else "penalized_path"),
                    lambda_index=(None if unpenalized else lambda_index),
                    lambda_fraction=(None if unpenalized else float(fractions[lambda_index])),
                    lambda_value=(None if unpenalized else float(path["lambda_max"] * fractions[lambda_index])),
                    lambda_max=(None if unpenalized else path["lambda_max"]), penalty_start=penalty_start)
                if (report["penalty_start"] != penalty_start
                        or len(report["terminal_beta"]) != (penalty_start if unpenalized else diagnostics["input_dim"] + 1)
                        or report["penalty"] != (0. if unpenalized else float(path["lambda_max"] * fractions[lambda_index]))
                        or report["outcome"] != "stationary"):
                    raise ValueError("pooled linear penalty/stationarity mismatch")
                _validate_report(report, config, policy, expected_context, RECOVERY_SCHEMA_V2)
        final = paths[-1]["fits"][-1]
        if (diagnostics["final_fit"] != final
                or diagnostics["refit_lambda_max"] != paths[-1]["lambda_max"]):
            raise ValueError("pooled final fit evidence mismatch")
        beta = np.asarray(arrays["standardized_coefficients"], dtype=float)
        coefficient = np.asarray(arrays["coefficients"], dtype=float)
        means, scales = (np.asarray(arrays[key], dtype=float) for key in ("means", "scales"))
        intercept = np.asarray(arrays["intercept"])
        if (beta.ndim != 1 or len(beta) < 2 or coefficient.shape != (len(beta) - 1,)
                or means.shape != coefficient.shape or scales.shape != coefficient.shape
                or np.any(scales <= 0) or intercept.shape != ()
                or not np.array_equal(means, np.asarray(paths[-1]["means"]))
                or not np.array_equal(scales, np.asarray(paths[-1]["scales"]))
                or not np.array_equal(beta, np.asarray(final["terminal_beta"]))
                or not np.array_equal(coefficient, beta[1:] / scales)
                or float(intercept) != float(beta[0] - means @ coefficient)
                or diagnostics["model_sha256"] != array_hash(np.r_[intercept, coefficient])):
            raise ValueError("pooled linear parameter transformation mismatch")
    else:
        from effect_modifier_experiment.survival.fisher import FisherConfig, _shapes
        options = _options(method_config)
        options.setdefault("architecture", family)
        config = FisherConfig(**options)
        config.validate()
        if set(arrays) != {"vector", "initial_vector"}:
            raise ValueError("invalid pooled neural array fields")
        vector = np.asarray(arrays["vector"])
        initial = np.asarray(arrays["initial_vector"])
        input_dim = int(diagnostics.get("input_dim", 0))
        expected_updates = config.epochs * math.ceil(sum(site_sizes) / config.batch_size)
        if (config.architecture != family or diagnostics["optimizer_config"] != asdict(config)
                or diagnostics.get("objective") != "half_mean_squared_error_divided_by_working_variance"
                or diagnostics.get("epochs") != config.epochs
                or diagnostics.get("optimizer_updates") != expected_updates
                or diagnostics.get("processed_rows") != config.epochs * sum(site_sizes)
                or diagnostics.get("fisher_computed") is not False
                or diagnostics.get("stop_reason") != "fixed_epoch_budget"
                or diagnostics.get("initial_eta") != initial_eta
                or diagnostics.get("training_seed") != seed + 1009
                or vector.ndim != 1 or input_dim < 5
                or input_dim % 2 != 1
                or initial.shape != vector.shape or initial[-1] != initial_eta
                or diagnostics["initial_sha256"] != sha256(initial.astype("<f8").tobytes()).hexdigest()
                or len(vector) != sum(a * b + a for a, b in _shapes(input_dim, family))
                or diagnostics["model_sha256"] != array_hash(vector)):
            raise ValueError("pooled neural optimizer/parameter mismatch")
        for key in ("initial_loss", "final_loss"):
            if not math.isfinite(diagnostics[key]) or diagnostics[key] < 0:
                raise ValueError("invalid pooled neural objective")
    if (not math.isfinite(diagnostics["total_fit_seconds"])
            or diagnostics["total_fit_seconds"] < 0):
        raise ValueError("invalid pooled fit timing")


def validate_pooled_fit(fit, fitted, family, scope, method_config, site_sizes):
    """Compatibility entrypoint for the experiment's immutable artifact store."""
    validate_fit(fit, fitted, family=family, pseudo_scope=scope,
                 method_config=method_config, seed=fit.get("seed"),
                 site_sizes=site_sizes, initial_eta=fit.get("initial_eta"))
