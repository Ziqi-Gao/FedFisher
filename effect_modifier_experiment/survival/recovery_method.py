"""Explicit numerical policies and saved evidence for dsLasso recovery.

The existing dsLasso pipeline owns preprocessing, CV, refitting and the proximal
updates. This observer adds guarded evaluations, a separate step cap and bounded
diagnostics only when explicitly requested. It owns no paths or scheduler state.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np

from .dslasso import (DslassoConfig, _Communication, _residuals, fit_dslasso,
                      validate_fit_diagnostics)
from .links import survival_mean, survival_mean_derivative


RECOVERY_SCHEMA_V1 = "survival_dslasso_numerical_recovery_v1"
RECOVERY_SCHEMA_V2 = "survival_dslasso_numerical_recovery_v2"


@dataclass(frozen=True)
class RecoveryPolicy:
    max_step: float = 1.0
    trace_every: int = 100
    trace_tail: int = 16
    saturation_epsilon: float = 1e-8

    def validate(self, config):
        config.validate()
        if (not np.isfinite(self.max_step) or self.max_step < config.initial_step
                or type(self.trace_every) is not int or self.trace_every < 1
                or type(self.trace_tail) is not int or self.trace_tail < 1
                or not np.isfinite(self.saturation_epsilon)
                or not 0 < self.saturation_epsilon < .5):
            raise ValueError("invalid recovery step/diagnostic policy")


class OptimizationFailure(RuntimeError):
    """A numerical endpoint, never a scientific model completion marker."""

    def __init__(self, reason, diagnostics, arrays):
        self.reason = reason
        self.diagnostics = diagnostics
        self.arrays = arrays
        super().__init__("survival LASSO recovery: " + reason)


class _RejectedEvaluation(Exception):
    pass


class _RecoveryCommunication(_Communication):
    def __init__(self):
        super().__init__()
        self.incomplete_evaluations = []

    def record(self, kind, num_sites, upload, download):
        super().record(kind, num_sites, upload, download)
        self.phases[kind]["attempted_rounds"] = self.phases[kind]["rounds"]

    def begin_evaluation(self, num_sites, width):
        phase = self.phases.setdefault("loss_gradient", dict(rounds=0,
            attempted_rounds=0, site_messages=0, upload_bytes=0, download_bytes=0))
        phase["attempted_rounds"] += 1
        phase["download_bytes"] += num_sites * 8 * width
        return phase

    def report(self):
        report = super().report()
        report["attempted_aggregation_rounds"] = sum(
            v["attempted_rounds"] for v in self.phases.values())
        report["incomplete_evaluations"] = list(self.incomplete_evaluations)
        return report


def _array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class _RecoveryRun:
    def __init__(self, config, policy, context):
        policy.validate(config)
        self.config, self.policy = config, policy
        self.context = dict(context or {})
        self.communication = _RecoveryCommunication()
        self.paths = []
        self.fit_context = {}
        self.calls = 0
        self.observer_seconds = 0.
        self.started = time.perf_counter(), time.process_time()
        self.stage = None
        self.stage_times = {key: None for key in ("cross_validation", "refit")}
        self.stage_cpu = dict(self.stage_times)
        self.stage_status = {key: "not_started" for key in self.stage_times}
        self.validation_progress = []
        self.current = None

    def initialize(self, config, site_sizes, fractions, assignments, sites):
        self.site_sizes = site_sizes
        self.fractions = fractions.tolist()
        before = time.perf_counter()
        self.fold_assignments_sha256 = _array_digest(*assignments)
        self.design_width = sites[0].rows.design.shape[1]
        self.site_folds = [dict(site_id=site.site_id,
            row_ids_sha256=_array_digest(site.rows.row_ids),
            assignments_sha256=_array_digest(labels),
            validation_sizes=np.bincount(labels, minlength=config.num_folds).tolist())
            for site, labels in zip(sites, assignments)]
        self.observer_seconds += time.perf_counter() - before

    def begin_stage(self, stage):
        self._close_stage("complete")
        self.stage = stage
        self.stage_started = time.perf_counter(), time.process_time()
        self.stage_status[stage] = "running"

    def _close_stage(self, status):
        if self.stage is not None:
            self.stage_times[self.stage] = time.perf_counter() - self.stage_started[0]
            self.stage_cpu[self.stage] = time.process_time() - self.stage_started[1]
            self.stage_status[self.stage] = status
            self.stage = None

    def begin_path(self, stage, fold, prepared, means, scales, rows=None):
        before = time.perf_counter()
        self.path = dict(stage=stage, fold=fold, lambda_max=None,
            partition_sha256=_array_digest(*(item for pair in prepared for item in pair)),
            means=np.asarray(means).tolist(), scales=np.asarray(scales).tolist(),
            partition_site_sizes=[len(y) for _, y in prepared],
            pseudo_values_sha256=_array_digest(*(y for _, y in prepared)),
            standardization_sha256=_array_digest(means, scales),
            row_ids_sha256=None if rows is None else _array_digest(*(r.row_ids for r in rows)),
            validation_partitions=[], unpenalized_start=None, fits=[])
        self.paths.append(self.path)
        self.observer_seconds += time.perf_counter() - before

    def set_fit_context(self, phase, **values):
        self.fit_context = dict(self.context, stage=self.path["stage"], fold=self.path["fold"],
            phase=phase, lambda_index=None, lambda_fraction=None, lambda_value=None,
            lambda_max=self.path["lambda_max"])
        self.fit_context.update(values)

    def set_lambda_max(self, value):
        if not np.isfinite(value) or value < 0:
            self.fail("nonfinite_lambda_max")
        self.path["lambda_max"] = value

    def completed_validation(self, fold, squared_error, validation_n):
        if not np.isfinite(squared_error).all():
            self.fail("nonfinite_validation_loss")
        self.validation_progress.append(dict(fold=fold,
            cumulative_squared_error=squared_error.tolist(), cumulative_n=int(validation_n)))

    def validation_partition(self, site_index, rows, design, target):
        before = time.perf_counter()
        self.path["validation_partitions"].append(dict(site_index=site_index, n=len(rows),
            row_ids_sha256=_array_digest(rows.row_ids),
            pseudo_values_sha256=_array_digest(target), standardized_design_sha256=_array_digest(design)))
        self.observer_seconds += time.perf_counter() - before

    def check_validation_predictor(self, eta):
        if not np.isfinite(eta).all():
            self.fail("nonfinite_validation_predictor")

    def start_fit(self, beta, penalty_start, penalty, config):
        self.fit_started = time.perf_counter(), time.process_time()
        self.fit_calls = self.calls
        self.current = dict(iterations=0, beta=beta.copy(), gradient=None,
            smooth=None, step=config.initial_step, backtracks=0)
        self.penalty_start, self.penalty = penalty_start, float(penalty)
        self.initial_smooth = self.initial_objective = None
        self.initial_l1 = float(np.abs(beta[penalty_start:]).sum())
        self.rejections = Counter()
        self.local_rejections = 0
        self.steps = []
        self.majorization_max = None
        self.majorization_gap_witness = self.majorization_guard_witness = None
        self.trace_fixed = {}
        self.trace_tail = deque(maxlen=self.policy.trace_tail)
        self.accepted_saturation = None

    def evaluate(self, sites, beta, communication=None):
        """Identical finite arithmetic to v1, with actual partial-round counts."""
        self.calls += 1
        phase = self.communication.begin_evaluation(len(sites), len(beta))
        loss_sum, gradient_sum, total = 0., np.zeros_like(beta), 0
        low = high = 0
        completed = 0
        try:
            for index, (design, response) in enumerate(sites):
                design = np.asarray(design, dtype=np.float64)
                response = np.asarray(response, dtype=np.float64)
                if (design.ndim != 2 or design.shape[1] != len(beta) or not len(design)
                        or response.shape != (len(design),) or not np.isfinite(design).all()
                        or not np.isfinite(response).all()):
                    raise ValueError("invalid local design/pseudo-response")
                eta = design @ beta
                if not np.isfinite(eta).all():
                    raise _RejectedEvaluation("nonfinite_predictor")
                mean = survival_mean(eta)
                residual = mean - response
                local_loss = .5 * float(residual @ residual)
                local_gradient = design.T @ (residual * survival_mean_derivative(eta))
                if not np.isfinite(local_loss) or not np.isfinite(local_gradient).all():
                    raise _RejectedEvaluation("nonfinite_site_loss_or_gradient")
                loss_sum += local_loss
                gradient_sum += local_gradient
                total += len(design)
                low += int(np.count_nonzero(mean <= self.policy.saturation_epsilon))
                high += int(np.count_nonzero(mean >= 1 - self.policy.saturation_epsilon))
                completed += 1
                phase["site_messages"] += 1
                phase["upload_bytes"] += 8 * (len(beta) + 2)
            if not np.isfinite(loss_sum) or not np.isfinite(gradient_sum).all():
                raise _RejectedEvaluation("nonfinite_aggregate")
        except _RejectedEvaluation as error:
            self.communication.incomplete_evaluations.append(dict(
                call=self.calls, completed_sites=completed, failed_site_index=(
                    completed if completed < len(sites) else None), reason=str(error),
                width=len(beta), total_sites=len(sites)))
            # The initial endpoint and lambda-max evaluation cannot be rejected
            # by a line search; their callers turn this into a structured failure.
            raise
        phase["rounds"] += 1
        self.last_saturation = dict(n=total, near_zero=low, near_one=high,
            near_zero_fraction=low / total, near_one_fraction=high / total,
            epsilon=self.policy.saturation_epsilon)
        return loss_sum / total, gradient_sum / total

    def accept_initial(self, beta, smooth, gradient, objective):
        self.initial_smooth, self.initial_objective = float(smooth), float(objective)
        if not np.isfinite(objective):
            self.fail("nonfinite_initial_objective")
        self.accepted_saturation = self.last_saturation

    def observe(self, iteration, beta, smooth, gradient, step, kkt, mapping, threshold, backtracks):
        objective = float(smooth + self.penalty * np.abs(beta[self.penalty_start:]).sum())
        norm_inf, norm_l2 = float(np.max(np.abs(beta))), float(np.linalg.norm(beta))
        self.current = dict(iterations=iteration, beta=beta.copy(), gradient=gradient.copy(),
            smooth=float(smooth), step=float(step), kkt=float(kkt), mapping=float(mapping),
            threshold=float(threshold), backtracks=backtracks, objective=objective,
            norm_inf=norm_inf, norm_l2=norm_l2)
        if not np.isfinite([smooth, step, kkt, mapping, threshold, objective, norm_inf, norm_l2]).all():
            self.fail("nonfinite_state")
        point = dict(iteration=int(iteration), smooth_objective=float(smooth),
            objective=objective, kkt_residual_inf=float(kkt),
            proximal_gradient_mapping_inf=float(mapping), stationarity_threshold=float(threshold),
            step=float(step), backtracks=int(backtracks), coefficient_norm_inf=norm_inf)
        if iteration % self.policy.trace_every == 0:
            self.trace_fixed[iteration] = point
        self.trace_tail.append(point)

    def trial(self, sites, candidate, beta, smooth, gradient, trial, communication):
        reason = None
        candidate_smooth = candidate_gradient = None
        if not np.isfinite(candidate).all():
            reason = "nonfinite_candidate"
            self.local_rejections += 1
        else:
            try:
                candidate_smooth, candidate_gradient = self.evaluate(sites, candidate, communication)
            except _RejectedEvaluation as error:
                reason = str(error)
            if reason is None:
                delta = candidate - beta
                bound = smooth + gradient @ delta + (delta @ delta) / (2 * trial)
                if not np.isfinite(delta).all() or not np.isfinite(bound):
                    reason = "nonfinite_delta_or_bound"
                elif candidate_smooth > bound + 1e-14:
                    reason = "majorization"
                else:
                    gap = float(candidate_smooth - bound)
                    # Retain the exact operands of the runtime comparison.
                    # (candidate - bound) <= tolerance is not its floating-point
                    # equivalent at the acceptance boundary.
                    accepted_bound = float(bound + 1e-14)
                    excess = float(candidate_smooth - accepted_bound)
                    larger_gap = self.majorization_max is None or gap > self.majorization_max
                    larger_excess = (self.majorization_guard_witness is None or
                        excess > self.majorization_guard_witness["guard_excess"])
                    if larger_gap or larger_excess:
                        witness = dict(candidate_smooth=float(candidate_smooth),
                            bound=float(bound), accepted_bound=accepted_bound, guard_excess=excess)
                        if larger_gap:
                            self.majorization_gap_witness = witness
                        if larger_excess:
                            self.majorization_guard_witness = witness
                    self.majorization_max = gap if self.majorization_max is None else max(
                        self.majorization_max, gap)
                    self.steps.append(float(trial))
                    self.accepted_saturation = self.last_saturation
        if reason is not None:
            self.rejections[reason] += 1
        return reason is None, candidate_smooth, candidate_gradient

    def _report(self, report, outcome):
        state = self.current
        trace = dict(self.trace_fixed)
        trace.update({point["iteration"]: point for point in self.trace_tail})
        report.update(outcome=outcome, context=dict(self.fit_context),
            terminal_beta=state["beta"].tolist(), terminal_gradient=(
                None if state["gradient"] is None else state["gradient"].tolist()),
            penalty=self.penalty, penalty_start=self.penalty_start,
            coefficient_norm_inf=state.get("norm_inf"), coefficient_norm_l2=state.get("norm_l2"),
            initial_smooth_objective=self.initial_smooth, final_smooth_objective=state["smooth"],
            initial_l1_norm=self.initial_l1,
            initial_step_size=float(self.config.initial_step),
            first_accepted_step=self.steps[0] if self.steps else None,
            min_accepted_step=min(self.steps) if self.steps else None,
            max_accepted_step=max(self.steps) if self.steps else None,
            accepted_steps=len(self.steps), majorization_max_gap=self.majorization_max,
            majorization_evidence=dict(accepted_checks=len(self.steps),
                max_gap_witness=self.majorization_gap_witness,
                max_guard_witness=self.majorization_guard_witness),
            rejected_candidates=dict(self.rejections), server_local_rejections=self.local_rejections,
            loss_gradient_calls=self.calls - self.fit_calls,
            elapsed_wall_seconds=time.perf_counter() - self.fit_started[0],
            elapsed_cpu_seconds=time.process_time() - self.fit_started[1],
            trace=[trace[key] for key in sorted(trace)], saturation=self.accepted_saturation)
        return _finite_json(report)

    def _save_report(self, report):
        if self.fit_context["phase"] == "unpenalized_start":
            self.path["unpenalized_start"] = report
        else:
            self.path["fits"].append(report)

    def finish_fit(self, report):
        outcome = "stationary" if report["converged"] else "iteration_limit"
        # Backtracking preserves the last finite state and measures rejected
        # evaluations. A nonfinite site evaluation is nevertheless an unexpected
        # process failure, never an eligible completed diagnostic observation.
        if self.communication.incomplete_evaluations:
            outcome = "nonfinite_evaluation"
        report = self._report(report, outcome)
        self._save_report(report)
        if outcome != "stationary":
            self.fail(outcome, report=report)
        return report

    def fail(self, reason, report=None, backtracks=None):
        state = self.current
        if report is None:
            report = self._report(dict(iterations=state["iterations"], converged=False,
                convergence_target="first_order_stationarity_not_global_optimum",
                initial_objective=self.initial_objective, final_objective=state.get("objective"),
                kkt_residual_inf=state.get("kkt"), proximal_gradient_mapping_inf=state.get("mapping"),
                stationarity_threshold=state.get("threshold"), final_step_size=state["step"],
                backtracks=state["backtracks"] if backtracks is None else backtracks), reason)
            self._save_report(report)
        arrays = dict(terminal_beta=state["beta"].copy(), terminal_gradient=(
            np.asarray([]) if state["gradient"] is None else state["gradient"].copy()))
        raise OptimizationFailure(reason, dict(outcome=reason,
            failure=dict(reason=reason, context=dict(self.fit_context), report=report)), arrays)

    def finalize(self, diagnostics, success):
        self._close_stage("complete" if success else "failed")
        diagnostics.update(recovery_schema=RECOVERY_SCHEMA_V2,
            outcome="stationary" if success else diagnostics["outcome"],
            config=asdict(self.config), recovery_policy=asdict(self.policy), context=self.context,
            site_sizes=getattr(self, "site_sizes", []),
            lambda_fractions=getattr(self, "fractions", []), paths=self.paths,
            fold_assignments_sha256=getattr(self, "fold_assignments_sha256", None),
            design_width=getattr(self, "design_width", None), site_folds=getattr(self, "site_folds", []),
            validation_progress=self.validation_progress,
            loss_gradient_calls=self.calls, communication=self.communication.report(),
            timing_seconds=dict(self.stage_times, full_fit=time.perf_counter() - self.started[0]),
            timing_cpu_seconds=dict(self.stage_cpu, full_fit=time.process_time() - self.started[1]),
            stage_status=self.stage_status,
            diagnostic_fingerprint_seconds=self.observer_seconds,
            diagnostic_fingerprint_scope="array_fingerprints_only; all_observer_work_included_in_full_fit",
            timing_scope="all_preprocessing_cv_refit_and_observers_excludes_artifact_serialization")
        return _finite_json(diagnostics)


def fit_recovery_dslasso(sites, tau, seed=0, config=DslassoConfig(),
                        policy=RecoveryPolicy(), context=None):
    """Run the shared full CV/refit pipeline under an explicit recovery policy."""
    observer = _RecoveryRun(config, policy, context)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            model, diagnostics = fit_dslasso(sites, tau, seed, config, recovery=observer)
        except _RejectedEvaluation as error:
            try:
                observer.fail(str(error))
            except OptimizationFailure as failure:
                failure.diagnostics = observer.finalize(failure.diagnostics, False)
                raise
        except OptimizationFailure as failure:
            failure.diagnostics = observer.finalize(failure.diagnostics, False)
            raise
    return model, observer.finalize(diagnostics, True)


def _require_finite(value, name, nonnegative=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or (nonnegative and value < 0)):
        raise ValueError("invalid recovery " + name)


def _validate_majorization(report, schema):
    """Replay the original accepted comparison, preserving strict r1 validation."""
    accepted = report["accepted_steps"]
    if schema not in (RECOVERY_SCHEMA_V1, RECOVERY_SCHEMA_V2):
        raise ValueError("unknown recovery evidence schema")
    if accepted:
        _require_finite(report["majorization_max_gap"], "majorization", False)
    if schema == RECOVERY_SCHEMA_V1:
        if accepted and report["majorization_max_gap"] > 1e-14:
            raise ValueError("recovery accepted a majorization violation")
        return
    evidence = report.get("majorization_evidence")
    if (not isinstance(evidence, dict) or set(evidence) != {
            "accepted_checks", "max_gap_witness", "max_guard_witness"}
            or type(evidence["accepted_checks"]) is not int
            or evidence["accepted_checks"] != accepted):
        raise ValueError("invalid recovery majorization evidence coverage")
    if not accepted:
        if any(evidence[key] is not None for key in ("max_gap_witness", "max_guard_witness")):
            raise ValueError("majorization witness claimed without accepted steps")
        return
    witnesses = [evidence[key] for key in ("max_gap_witness", "max_guard_witness")]
    for witness in witnesses:
        if not isinstance(witness, dict) or set(witness) != {
                "candidate_smooth", "bound", "accepted_bound", "guard_excess"}:
            raise ValueError("invalid recovery majorization witness")
        for key, value in witness.items():
            _require_finite(value, "majorization " + key, key == "candidate_smooth")
        candidate, bound = witness["candidate_smooth"], witness["bound"]
        if (witness["accepted_bound"] != bound + 1e-14
                or witness["guard_excess"] != candidate - witness["accepted_bound"]
                or candidate > witness["accepted_bound"]):
            raise ValueError("recovery accepted a majorization violation or invalid witness")
    gap_witness, guard_witness = witnesses
    if (report["majorization_max_gap"] != gap_witness["candidate_smooth"] - gap_witness["bound"]
            or gap_witness["guard_excess"] > guard_witness["guard_excess"]
            or guard_witness["candidate_smooth"] - guard_witness["bound"] > report["majorization_max_gap"]):
        raise ValueError("recovery majorization extrema disagree with witnesses")


def _validate_report(report, config, policy, context, schema=RECOVERY_SCHEMA_V2):
    """Recompute endpoint arithmetic using saved arrays, without prediction."""
    beta = np.asarray(report["terminal_beta"], dtype=float)
    gradient = np.asarray(report["terminal_gradient"], dtype=float)
    start, penalty = report["penalty_start"], report["penalty"]
    if (beta.ndim != 1 or not len(beta) or gradient.shape != beta.shape
            or not np.isfinite(beta).all() or not np.isfinite(gradient).all()
            or type(start) is not int or not 0 <= start <= len(beta)
            or report["context"] != context):
        raise ValueError("invalid recovery terminal arrays/context")
    _require_finite(penalty, "penalty")
    step = report["final_step_size"]
    _require_finite(step, "step")
    if not config.min_step <= step <= policy.max_step:
        raise ValueError("recovery step outside policy")
    norm = float(np.max(np.abs(beta)))
    threshold = float(config.tolerance * (1 + norm))
    kkt, mapping = _residuals(beta, gradient, penalty, start, step)
    for key, expected in (("coefficient_norm_inf", norm),
            ("coefficient_norm_l2", float(np.linalg.norm(beta))),
            ("stationarity_threshold", threshold), ("kkt_residual_inf", kkt),
            ("proximal_gradient_mapping_inf", mapping)):
        if report[key] != expected or not np.isfinite(expected):
            raise ValueError("recovery terminal arithmetic mismatch: " + key)
    for key in ("initial_smooth_objective", "final_smooth_objective", "initial_objective",
                "final_objective", "initial_l1_norm", "elapsed_wall_seconds", "elapsed_cpu_seconds"):
        _require_finite(report[key], key)
    if (report["initial_objective"] != report["initial_smooth_objective"] + penalty * report["initial_l1_norm"]
            or report["final_objective"] != report["final_smooth_objective"] + penalty * np.abs(beta[start:]).sum()
            or report["final_objective"] > report["initial_objective"] + 1e-10):
        raise ValueError("recovery objective decomposition or monotonicity mismatch")
    iterations, accepted = report["iterations"], report["accepted_steps"]
    if (type(iterations) is not int or not 0 <= iterations <= config.max_iter
            or type(accepted) is not int or accepted != iterations
            or report["initial_step_size"] != config.initial_step
            or report["convergence_target"] != "first_order_stationarity_not_global_optimum"):
        raise ValueError("invalid recovery iterations")
    stationary = kkt <= threshold and mapping <= threshold
    if report["outcome"] == "stationary":
        if report["converged"] is not True or not stationary:
            raise ValueError("false recovery stationarity")
    elif report["outcome"] == "iteration_limit":
        if report["converged"] is not False or stationary or iterations != config.max_iter:
            raise ValueError("invalid recovery iteration-limit evidence")
    else:
        raise ValueError("unexpected numerical failure cannot complete diagnostic")
    if accepted:
        if not (config.min_step <= report["min_accepted_step"] <= step <=
                report["max_accepted_step"] <= policy.max_step
                and config.min_step <= report["first_accepted_step"] <= config.initial_step):
            raise ValueError("invalid recovery accepted step bounds")
    elif any(report[key] is not None for key in (
            "first_accepted_step", "min_accepted_step", "max_accepted_step", "majorization_max_gap")):
        raise ValueError("steps claimed without iterations")
    _validate_majorization(report, schema)
    rejected = report["rejected_candidates"]
    for count in list(rejected.values()) + [report["backtracks"], report["server_local_rejections"],
                                          report["loss_gradient_calls"]]:
        if type(count) is not int or count < 0:
            raise ValueError("invalid recovery evaluation count")
    if (sum(rejected.values()) != report["backtracks"]
            or report["server_local_rejections"] != rejected.get("nonfinite_candidate", 0)
            or report["loss_gradient_calls"] != 1 + iterations + report["backtracks"] - report["server_local_rejections"]):
        raise ValueError("recovery evaluation accounting mismatch")
    trace = report["trace"]
    expected_indices = sorted(set(range(0, iterations + 1, policy.trace_every)) |
                              set(range(max(0, iterations - policy.trace_tail + 1), iterations + 1)))
    if [point["iteration"] for point in trace] != expected_indices:
        raise ValueError("recovery bounded trace coverage mismatch")
    if (trace[0]["objective"] != report["initial_objective"]
            or trace[0]["smooth_objective"] != report["initial_smooth_objective"]
            or trace[-1]["smooth_objective"] != report["final_smooth_objective"]):
        raise ValueError("recovery trace objective endpoint mismatch")
    previous = report["initial_objective"]
    previous_backtracks = 0
    for point in trace:
        for key in ("objective", "smooth_objective", "kkt_residual_inf", "proximal_gradient_mapping_inf",
                    "stationarity_threshold", "step", "coefficient_norm_inf"):
            _require_finite(point[key], "trace " + key)
        if (point["objective"] > previous + 1e-10
                or point["stationarity_threshold"] != config.tolerance * (1 + point["coefficient_norm_inf"])
                or not config.min_step <= point["step"] <= policy.max_step
                or type(point["backtracks"]) is not int
                or not previous_backtracks <= point["backtracks"] <= report["backtracks"]):
            raise ValueError("invalid recovery trace arithmetic")
        previous = point["objective"]
        previous_backtracks = point["backtracks"]
    for trace_key, report_key in (("objective", "final_objective"), ("step", "final_step_size"),
            ("kkt_residual_inf", "kkt_residual_inf"),
            ("proximal_gradient_mapping_inf", "proximal_gradient_mapping_inf"),
            ("stationarity_threshold", "stationarity_threshold"), ("backtracks", "backtracks"),
            ("coefficient_norm_inf", "coefficient_norm_inf")):
        if trace[-1][trace_key] != report[report_key]:
            raise ValueError("recovery trace endpoint mismatch")
    saturation = report["saturation"]
    if (type(saturation["n"]) is not int or saturation["n"] < 1
            or saturation["epsilon"] != policy.saturation_epsilon):
        raise ValueError("invalid recovery saturation scope")
    for key in ("near_zero", "near_one"):
        if (type(saturation[key]) is not int or not 0 <= saturation[key] <= saturation["n"]
                or saturation[key + "_fraction"] != saturation[key] / saturation["n"]):
            raise ValueError("invalid recovery saturation counts")
    if saturation["near_zero"] + saturation["near_one"] > saturation["n"]:
        raise ValueError("overlapping recovery saturation categories")


def validate_recovery_diagnostics(diagnostics, config, policy, arrays, site_sizes,
                                  require_success=None, required_schema=None):
    """Validate saved successful fits or complete iteration-limit observations."""
    cfg = config if isinstance(config, DslassoConfig) else DslassoConfig(**config)
    pol = policy if isinstance(policy, RecoveryPolicy) else RecoveryPolicy(**policy)
    pol.validate(cfg)
    d = diagnostics
    outcome = d.get("outcome")
    schema = d.get("recovery_schema")
    if (schema not in (RECOVERY_SCHEMA_V1, RECOVERY_SCHEMA_V2)
            or (required_schema is not None and schema != required_schema)
            or d.get("config") != asdict(cfg) or d.get("recovery_policy") != asdict(pol)
            or d.get("site_sizes") != list(site_sizes)
            or outcome not in ("stationary", "iteration_limit")
            or (require_success is True and outcome != "stationary")
            or (require_success is False and outcome != "iteration_limit")):
        raise ValueError("recovery fit identity/outcome mismatch")
    fractions = np.geomspace(1., cfg.lambda_min_ratio, cfg.num_lambdas)
    if (not np.array_equal(d["lambda_fractions"], fractions)
            or not isinstance(d["fold_assignments_sha256"], str)
            or len(d["fold_assignments_sha256"]) != 64):
        raise ValueError("invalid recovery fold/grid evidence")
    paths = d["paths"]
    if not 1 <= len(paths) <= cfg.num_folds + 1:
        raise ValueError("invalid recovery path count")
    width = d["design_width"]
    if type(width) is not int or width < 5 or width % 2 != 1 or len(d["site_folds"]) != len(site_sizes):
        raise ValueError("invalid recovery design/site dimensions")
    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if not valid_hash(d["fold_assignments_sha256"]):
        raise ValueError("invalid recovery assignment hash")
    site_ids = []
    for n, site in zip(site_sizes, d["site_folds"]):
        site_ids.append(site["site_id"])
        if (type(site["site_id"]) is not int or site["site_id"] < 0
                or site["validation_sizes"] != [(n + cfg.num_folds - 1 - fold) // cfg.num_folds
                                                 for fold in range(cfg.num_folds)]
                or not valid_hash(site["row_ids_sha256"]) or not valid_hash(site["assignments_sha256"])):
            raise ValueError("invalid recovery deterministic fold sizes/identity")
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("duplicate recovery site identity")
    all_reports = []
    expected_calls = 0
    expected_upload = expected_download = 0
    for index, path in enumerate(paths):
        stage, fold = ("cv", index) if index < cfg.num_folds else ("refit", None)
        expected_sizes = list(site_sizes) if stage == "refit" else [
            n - site["validation_sizes"][fold] for n, site in zip(site_sizes, d["site_folds"])]
        if (path["stage"] != stage or path["fold"] != fold
                or path["partition_site_sizes"] != expected_sizes
                or any(not valid_hash(path[key]) for key in ("partition_sha256", "pseudo_values_sha256",
                                                             "standardization_sha256", "row_ids_sha256"))):
            raise ValueError("invalid recovery path identity")
        means, scales = np.asarray(path["means"]), np.asarray(path["scales"])
        if (means.shape != (width,) or scales.shape != means.shape
                or not np.isfinite(means).all() or not np.isfinite(scales).all()
                or np.any((scales < cfg.scale_floor) & (scales != 1.))
                or path["standardization_sha256"] != _array_digest(means, scales)):
            raise ValueError("invalid recovery normalization evidence")
        reports = [path["unpenalized_start"]] + path["fits"]
        if reports[0] is None or len(path["fits"]) > cfg.num_lambdas:
            raise ValueError("missing recovery start/path")
        if path["lambda_max"] is not None:
            _require_finite(path["lambda_max"], "lambda maximum")
            expected_calls += 1
            expected_upload += len(site_sizes) * 8 * (width + 3)
            expected_download += len(site_sizes) * 8 * (width + 1)
        for fit_index, report in enumerate(reports):
            context = dict(d["context"], stage=stage, fold=fold,
                phase="unpenalized_start" if fit_index == 0 else "penalized_path",
                lambda_index=None if fit_index == 0 else fit_index - 1,
                lambda_fraction=None if fit_index == 0 else float(fractions[fit_index - 1]),
                lambda_value=None if fit_index == 0 else float(path["lambda_max"] * fractions[fit_index - 1]),
                lambda_max=None if fit_index == 0 else path["lambda_max"],
                penalty_start=(len(means) - 1) // 2 + 2)
            if (report["penalty_start"] != context["penalty_start"]
                    or report["penalty"] != (0. if fit_index == 0 else context["lambda_value"])
                    or len(report["terminal_beta"]) != (context["penalty_start"] if fit_index == 0 else width + 1)):
                raise ValueError("recovery penalty metadata mismatch")
            _validate_report(report, cfg, pol, context, schema)
            if report["saturation"]["n"] != sum(path["partition_site_sizes"]):
                raise ValueError("recovery saturation partition mismatch")
            all_reports.append(report)
            expected_calls += report["loss_gradient_calls"]
            expected_upload += report["loss_gradient_calls"] * len(site_sizes) * 8 * (len(report["terminal_beta"]) + 2)
            expected_download += report["loss_gradient_calls"] * len(site_sizes) * 8 * len(report["terminal_beta"])
        if index < len(paths) - 1 and (len(path["fits"]) != cfg.num_lambdas
                or any(report["outcome"] != "stationary" for report in reports)):
            raise ValueError("recovery continued past incomplete CV path")
    progress = d["validation_progress"]
    completed_folds = cfg.num_folds if paths[-1]["stage"] == "refit" else len(paths) - 1
    if len(progress) != completed_folds:
        raise ValueError("recovery validation progress coverage mismatch")
    previous_error, cumulative_n = np.zeros(cfg.num_lambdas), 0
    for fold, record in enumerate(progress):
        partitions = paths[fold]["validation_partitions"]
        if len(partitions) != len(site_sizes):
            raise ValueError("incomplete recovery validation partitions")
        for site_index, partition in enumerate(partitions):
            n = d["site_folds"][site_index]["validation_sizes"][fold]
            if (partition["site_index"] != site_index or partition["n"] != n
                    or any(not valid_hash(partition[key]) for key in (
                        "row_ids_sha256", "pseudo_values_sha256", "standardized_design_sha256"))):
                raise ValueError("invalid recovery validation partition")
            cumulative_n += n
        error = np.asarray(record["cumulative_squared_error"])
        if (record["fold"] != fold or record["cumulative_n"] != cumulative_n
                or error.shape != (cfg.num_lambdas,) or not np.isfinite(error).all()
                or np.any(error < previous_error)):
            raise ValueError("invalid recovery cumulative CV loss")
        previous_error = error
    if paths[-1]["stage"] == "cv" and paths[-1]["validation_partitions"]:
        raise ValueError("recovery failed path unexpectedly validated")
    if paths[-1]["stage"] == "refit":
        if paths[-1]["validation_partitions"]:
            raise ValueError("refit cannot contain validation partitions")
        selected = int(np.argmin(previous_error / cumulative_n))
        if len(paths[-1]["fits"]) > selected + 1:
            raise ValueError("recovery refit ran beyond selected fraction")
        if outcome == "stationary" and (len(paths[-1]["fits"]) != selected + 1
                or d["selected_lambda_index"] != selected
                or not np.array_equal(d["cv_loss"], previous_error / cumulative_n)):
            raise ValueError("recovery CV loss/refit selection mismatch")
    if (type(d["loss_gradient_calls"]) is not int or d["loss_gradient_calls"] != expected_calls
            or any(report["outcome"] != "stationary" for report in all_reports[:-1])):
        raise ValueError("recovery calls or earlier stationarity mismatch")
    if outcome == "stationary":
        validate_fit_diagnostics(d, cfg, arrays, site_sizes)
        if len(paths) != cfg.num_folds + 1 or all_reports[-1]["outcome"] != "stationary":
            raise ValueError("incomplete recovery successful fit")
        for index, path in enumerate(paths[:-1]):
            if (path["unpenalized_start"] != d["cv_paths"][index]["unpenalized_start"]
                    or path["fits"] != d["cv_paths"][index]["fits"]
                    or path["lambda_max"] != d["cv_paths"][index]["lambda_max"]):
                raise ValueError("recovery duplicated CV evidence differs")
        if (paths[-1]["unpenalized_start"] != d["refit_start"] or paths[-1]["fits"] != d["refit_path"]
                or paths[-1]["lambda_max"] != d["refit_lambda_max"]):
            raise ValueError("recovery duplicated refit evidence differs")
        beta = np.asarray(all_reports[-1]["terminal_beta"])
        coefficients = beta[1:] / np.asarray(paths[-1]["scales"])
        intercept = beta[0] - np.asarray(paths[-1]["means"]) @ coefficients
        if (not np.array_equal(coefficients, arrays["coefficients"])
                or float(np.asarray(arrays["intercept"]).reshape(-1)[0]) != intercept):
            raise ValueError("recovery model does not match terminal refit")
    else:
        report = all_reports[-1]
        if (report["outcome"] != "iteration_limit" or d["failure"] != dict(
                reason="iteration_limit", context=report["context"], report=report)
                or not np.array_equal(arrays["terminal_beta"], report["terminal_beta"])
                or not np.array_equal(arrays["terminal_gradient"], report["terminal_gradient"])):
            raise ValueError("recovery failure endpoint mismatch")
    communication = d["communication"]
    phases = communication["phases"]
    if (communication["one_shot"] is not False or communication["accounting"] !=
            "float64_numeric_payload_only_no_transport_or_serialization"
            or not {"standardization", "response_initialization", "loss_gradient"} <= set(phases)
            or set(phases) - {"standardization", "response_initialization", "loss_gradient", "validation_path_loss"}
            or phases["loss_gradient"]["attempted_rounds"] != expected_calls
            or phases["loss_gradient"]["rounds"] != expected_calls
            or phases["loss_gradient"]["upload_bytes"] != expected_upload
            or phases["loss_gradient"]["download_bytes"] != expected_download
            or communication["incomplete_evaluations"]):
        raise ValueError("invalid recovery complete communication")
    for name, rounds, upload, download in (
            ("standardization", len(paths), 8 + 16 * width, 16 * width),
            ("response_initialization", len(paths), 16, 8),
            ("validation_path_loss", completed_folds, 8 * (1 + cfg.num_lambdas), 8 * (width + 1) * cfg.num_lambdas)):
        phase = phases.get(name)
        if rounds == 0 and phase is None:
            continue
        if (phase is None or phase["rounds"] != rounds
                or phase["upload_bytes"] != rounds * len(site_sizes) * upload
                or phase["download_bytes"] != rounds * len(site_sizes) * download):
            raise ValueError("recovery phase payload accounting mismatch")
    for phase in phases.values():
        for key in ("rounds", "attempted_rounds", "site_messages", "upload_bytes", "download_bytes"):
            if type(phase[key]) is not int or phase[key] < 0:
                raise ValueError("invalid recovery communication count")
        if phase["attempted_rounds"] != phase["rounds"] or phase["site_messages"] != phase["rounds"] * len(site_sizes):
            raise ValueError("invalid recovery complete site counts")
    for total, field in (("aggregation_rounds", "rounds"), ("attempted_aggregation_rounds", "attempted_rounds"),
                          ("upload_bytes", "upload_bytes"), ("download_bytes", "download_bytes")):
        if communication[total] != sum(phase[field] for phase in phases.values()):
            raise ValueError("recovery communication totals mismatch")
    for time_key in ("timing_seconds", "timing_cpu_seconds"):
        times = d[time_key]
        if set(times) != {"cross_validation", "refit", "full_fit"}:
            raise ValueError("invalid recovery timing keys")
        _require_finite(times["full_fit"], "full timing")
        for stage in ("cross_validation", "refit"):
            status = d["stage_status"][stage]
            if times[stage] is None:
                if status != "not_started" or stage != "refit" or outcome == "stationary":
                    raise ValueError("missing recovery executed timing")
            else:
                _require_finite(times[stage], "stage timing")
                if status not in ("complete", "failed"):
                    raise ValueError("invalid recovery stage completion")
        if sum(value for key, value in times.items() if key != "full_fit" and value is not None) > times["full_fit"] + 1e-6:
            raise ValueError("recovery stages exceed total timing")
        report_key = "elapsed_wall_seconds" if time_key == "timing_seconds" else "elapsed_cpu_seconds"
        for stage, name in (("cv", "cross_validation"), ("refit", "refit")):
            measured = sum(report[report_key] for report in all_reports if report["context"]["stage"] == stage)
            if measured > (times[name] or 0.) + 1e-6:
                raise ValueError("recovery optimizer times exceed enclosing stage")
    expected_status = dict(cross_validation="complete", refit="complete")
    if outcome == "iteration_limit":
        expected_status = (dict(cross_validation="failed", refit="not_started")
                           if paths[-1]["stage"] == "cv" else dict(cross_validation="complete", refit="failed"))
    if d["stage_status"] != expected_status:
        raise ValueError("recovery stage status does not match outcome")
    _require_finite(d["diagnostic_fingerprint_seconds"], "fingerprint timing")
    if (d["diagnostic_fingerprint_seconds"] > d["timing_seconds"]["full_fit"]
            or d["diagnostic_fingerprint_scope"] != "array_fingerprints_only; all_observer_work_included_in_full_fit"):
        raise ValueError("observer timing exceeds total")
    return True
