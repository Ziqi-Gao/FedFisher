import copy
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

import numpy as np

from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite
from effect_modifier_experiment.survival.dslasso import DslassoConfig, fit_dslasso, proximal_fit
from effect_modifier_experiment.survival.recovery_method import (
    OptimizationFailure, RecoveryPolicy, RECOVERY_SCHEMA_V1, RECOVERY_SCHEMA_V2,
    _RecoveryRun, _validate_report,
    fit_recovery_dslasso, validate_recovery_diagnostics,
)


def tiny_sites(nonzero=False):
    sites = []
    for site_id in range(2):
        n = 32
        x = np.zeros((n, 2))
        treatment = np.zeros(n)
        if nonzero:
            rng = np.random.default_rng(42 + site_id)
            x = .15 * rng.normal(size=(n, 2))
            treatment = np.tile([0., 1.], n // 2)
        rows = SurvivalRows(x, treatment, np.tile([1., 2., 4., 5.], n // 4),
                            np.ones(n), np.arange(site_id * n, (site_id + 1) * n))
        sites.append(SurvivalSite(site_id, rows))
    return sites


def kernel_observer(sites, cfg, policy):
    observer = _RecoveryRun(cfg, policy, {"scenario": "fixture", "seed": 0, "recipe": "fixture"})
    observer.begin_stage("cross_validation")
    width = sites[0][0].shape[1]
    observer.begin_path("cv", 0, sites, np.zeros(width - 1), np.ones(width - 1))
    observer.set_fit_context("unpenalized_start", penalty_start=width)
    return observer


class RecoveryMethodTests(unittest.TestCase):
    def test_old_config_schema_has_no_recovery_fields(self):
        self.assertEqual(set(asdict(DslassoConfig())), {
            "num_folds", "num_lambdas", "lambda_min_ratio", "max_iter", "tolerance",
            "initial_step", "min_step", "scale_floor"})
        for policy in (RecoveryPolicy(max_step=.5), RecoveryPolicy(max_step=np.inf),
                       RecoveryPolicy(trace_every=0), RecoveryPolicy(trace_tail=0)):
            with self.assertRaises(ValueError):
                policy.validate(DslassoConfig())

    def test_baseline_kernel_exact_with_penalty_and_warm_start(self):
        x = np.array([[1., -1.], [1., -.3], [1., .4], [1., 1.2]])
        y = np.array([-.1, .3, .9, 1.1])
        sites = [(x[:1], y[:1]), (x[1:], y[1:])]
        cfg = DslassoConfig(max_iter=3000, tolerance=1e-7)
        initial = np.array([-.2, .1])
        for penalty in (.04, .02):
            expected, old = proximal_fit(sites, initial, 1, penalty, cfg)
            observer = kernel_observer(sites, cfg, RecoveryPolicy())
            observer.set_fit_context("penalized_path", penalty_start=1, lambda_index=0,
                                     lambda_fraction=1., lambda_value=penalty)
            actual, new = proximal_fit(sites, initial, 1, penalty, cfg,
                                       observer.communication, observer)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual({key: new[key] for key in old}, old)
            _validate_report(new, cfg, observer.policy, observer.fit_context)
            initial = expected

    def test_full_cv_baseline_exact_and_saved_validation(self):
        sites = tiny_sites(True)
        cfg = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=4000, tolerance=1e-5)
        old_model, old = fit_dslasso(sites, 3., config=cfg)
        model, new = fit_recovery_dslasso(sites, 3., config=cfg)
        np.testing.assert_array_equal(old_model.coefficients, model.coefficients)
        self.assertEqual(old_model.intercept, model.intercept)
        for key in ("cv_loss", "selected_lambda_index", "selected_lambda_value", "lambda_fractions"):
            self.assertEqual(old[key], new[key])
        arrays = dict(intercept=np.array(model.intercept), coefficients=model.coefficients)
        validate_recovery_diagnostics(new, cfg, RecoveryPolicy(), arrays, [32, 32], True)
        for key in ("aggregation_rounds", "upload_bytes", "download_bytes"):
            self.assertEqual(old["communication"][key], new["communication"][key])
        for field in ("stationarity_threshold", "kkt_residual_inf", "final_objective"):
            bad = copy.deepcopy(new)
            bad["paths"][-1]["fits"][-1][field] += .01
            with self.assertRaises(ValueError):
                validate_recovery_diagnostics(bad, cfg, RecoveryPolicy(), arrays, [32, 32])

    def test_larger_cap_accepts_larger_steps(self):
        sites = [(np.ones((4, 1)) * .1, np.array([0., 0., 1., 1.]))]
        cfg = DslassoConfig(max_iter=4000)
        observer = kernel_observer(sites, cfg, RecoveryPolicy(max_step=64))
        beta, report = proximal_fit(sites, np.zeros(1), 1, 0., cfg,
                                    observer.communication, observer)
        self.assertGreater(report["max_accepted_step"], 1.)
        self.assertLessEqual(report["max_accepted_step"], 64.)
        self.assertAlmostEqual(.1 * beta[0], np.log(-np.log(.5)), places=3)
        self.assertEqual(report["first_accepted_step"], 1.)
        _validate_report(report, cfg, observer.policy, observer.fit_context)

    def test_backtracking_majorization_and_counts(self):
        sites = [(np.ones((4, 1)) * 10, np.array([0., 0., 1., 1.]))]
        cfg = DslassoConfig(max_iter=1000, tolerance=1e-6)
        observer = kernel_observer(sites, cfg, RecoveryPolicy(max_step=64))
        _, report = proximal_fit(sites, np.zeros(1), 1, 0., cfg,
                                 observer.communication, observer)
        self.assertGreater(report["backtracks"], 0)
        self.assertLessEqual(report["majorization_evidence"]["max_guard_witness"]["guard_excess"], 0.)
        self.assertEqual(report["loss_gradient_calls"], 1 + report["iterations"] + report["backtracks"])
        _validate_report(report, cfg, observer.policy, observer.fit_context)

    def test_float_boundary_accepted_step_replays_exact_runtime_guard(self):
        sites = [(np.ones((1, 1)), np.zeros(1))]
        cfg = DslassoConfig(max_iter=1)
        observer = kernel_observer(sites, cfg, RecoveryPolicy())
        # One quadratic-like step: delta=-1, initial gradient=1 and smooth=.6.
        # The final loss lies exactly on the rounded runtime acceptance bound.
        bound = .6 - 1. + .5
        loss = float(bound + 1e-14)
        self.assertGreater(loss - bound, 1e-14)
        evaluations = iter(((.6, np.ones(1)), (loss, np.zeros(1))))

        def evaluate(*args):
            observer.calls += 1
            observer.last_saturation = dict(n=1, near_zero=0, near_one=0,
                near_zero_fraction=0., near_one_fraction=0., epsilon=1e-8)
            return next(evaluations)

        with patch.object(observer, "evaluate", side_effect=evaluate):
            beta, report = proximal_fit(sites, np.zeros(1), 1, 0., cfg,
                                        observer.communication, observer)
        np.testing.assert_array_equal(beta, [-1.])
        self.assertTrue(report["converged"])
        self.assertGreater(report["majorization_max_gap"], 1e-14)
        report = json.loads(json.dumps(report, allow_nan=False))
        _validate_report(report, cfg, observer.policy, observer.fit_context)
        # The archived validator remains strict; only new evidence repairs the
        # boundary rather than retrospectively weakening historical checks.
        with self.assertRaisesRegex(ValueError, "majorization violation"):
            _validate_report(report, cfg, observer.policy, observer.fit_context, RECOVERY_SCHEMA_V1)

        def check(edit):
            bad = copy.deepcopy(report)
            edit(bad)
            with self.assertRaises(ValueError):
                _validate_report(bad, cfg, observer.policy, observer.fit_context)

        check(lambda d: d.pop("majorization_evidence"))
        check(lambda d: d["majorization_evidence"].__setitem__("accepted_checks", 2))
        check(lambda d: d.__setitem__("majorization_max_gap", 0.))
        for name in ("max_gap_witness", "max_guard_witness"):
            for key in ("candidate_smooth", "bound", "accepted_bound", "guard_excess"):
                check(lambda d, name=name, key=key: d["majorization_evidence"][name].__setitem__(
                    key, float(np.nextafter(d["majorization_evidence"][name][key], np.inf))))

    def test_one_ulp_above_runtime_bound_is_still_rejected(self):
        sites = [(np.ones((1, 1)), np.zeros(1))]
        cfg = DslassoConfig()
        observer = kernel_observer(sites, cfg, RecoveryPolicy())
        observer.start_fit(np.zeros(1), 1, 0., cfg)
        bound = .1
        rejected_loss = float(np.nextafter(bound + 1e-14, np.inf))
        with patch.object(observer, "evaluate", return_value=(rejected_loss, np.zeros(1))):
            accepted, _, _ = observer.trial(sites, np.zeros(1), np.zeros(1),
                bound, np.zeros(1), 1., observer.communication)
        self.assertFalse(accepted)
        self.assertEqual(observer.rejections["majorization"], 1)
        self.assertEqual(observer.steps, [])
        self.assertIsNone(observer.majorization_gap_witness)
        self.assertIsNone(observer.majorization_guard_witness)

    def test_new_schema_is_required_without_disabling_archived_validation(self):
        cfg = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1000)
        model, diagnostics = fit_recovery_dslasso(tiny_sites(), 3., config=cfg)
        arrays = dict(intercept=np.array(model.intercept), coefficients=model.coefficients)
        self.assertEqual(diagnostics["recovery_schema"], RECOVERY_SCHEMA_V2)
        validate_recovery_diagnostics(diagnostics, cfg, RecoveryPolicy(), arrays, [32, 32],
                                      required_schema=RECOVERY_SCHEMA_V2)
        archived = copy.deepcopy(diagnostics)
        archived["recovery_schema"] = RECOVERY_SCHEMA_V1

        def remove_new_evidence(value):
            if isinstance(value, dict):
                value.pop("majorization_evidence", None)
                for item in value.values():
                    remove_new_evidence(item)
            elif isinstance(value, list):
                for item in value:
                    remove_new_evidence(item)

        remove_new_evidence(archived)
        validate_recovery_diagnostics(archived, cfg, RecoveryPolicy(), arrays, [32, 32],
                                      required_schema=RECOVERY_SCHEMA_V1)
        with self.assertRaises(ValueError):
            validate_recovery_diagnostics(archived, cfg, RecoveryPolicy(), arrays, [32, 32],
                                          required_schema=RECOVERY_SCHEMA_V2)
        archived["recovery_schema"] = RECOVERY_SCHEMA_V2
        with self.assertRaises(ValueError):
            validate_recovery_diagnostics(archived, cfg, RecoveryPolicy(), arrays, [32, 32])

    def test_budget_prefix_and_typed_endpoint(self):
        sites = [(np.ones((4, 1)), np.array([0., 0., 1., 1.]))]
        reports = []
        for budget in (2, 5):
            cfg = DslassoConfig(max_iter=budget, tolerance=1e-12)
            observer = kernel_observer(sites, cfg, RecoveryPolicy())
            with self.assertRaises(OptimizationFailure) as caught:
                proximal_fit(sites, np.ones(1), 1, 0., cfg, observer.communication, observer)
            error = caught.exception
            self.assertEqual(error.reason, "iteration_limit")
            report = error.diagnostics["failure"]["report"]
            self.assertEqual(report["iterations"], budget)
            np.testing.assert_array_equal(error.arrays["terminal_beta"], report["terminal_beta"])
            _validate_report(report, cfg, observer.policy, observer.fit_context)
            reports.append(report)
        self.assertEqual(reports[0]["trace"], reports[1]["trace"][:3])

    def test_line_search_exhaustion_is_distinct_and_finite(self):
        sites = [(np.ones((4, 1)) * 10, np.array([0., 0., 1., 1.]))]
        cfg = DslassoConfig(min_step=1.)
        observer = kernel_observer(sites, cfg, RecoveryPolicy())
        with self.assertRaises(OptimizationFailure) as caught:
            proximal_fit(sites, np.zeros(1), 1, 0., cfg, observer.communication, observer)
        self.assertEqual(caught.exception.reason, "line_search_exhausted")
        self.assertGreater(caught.exception.diagnostics["failure"]["report"]["backtracks"], 0)
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_nonfinite_predictor_rejected_without_fake_site_response(self):
        sites = [(np.ones((2, 1)), np.zeros(2)),
                 (np.ones((2, 1)) * 1e308, np.zeros(2))]
        observer = kernel_observer(sites, DslassoConfig(), RecoveryPolicy())
        observer.start_fit(np.array([10.]), 1, 0., DslassoConfig())
        with np.errstate(over="ignore"):
            accepted, _, _ = observer.trial(sites, np.array([10.]), np.zeros(1),
                                            .5, np.ones(1), 1., observer.communication)
        self.assertFalse(accepted)
        phase = observer.communication.report()["phases"]["loss_gradient"]
        self.assertEqual((phase["attempted_rounds"], phase["rounds"], phase["site_messages"]), (1, 0, 1))
        self.assertEqual(phase["upload_bytes"], 24)
        event = observer.communication.incomplete_evaluations[0]
        self.assertEqual((event["completed_sites"], event["failed_site_index"]), (1, 1))
        calls = observer.calls
        accepted, _, _ = observer.trial(sites, np.array([np.inf]), np.zeros(1),
                                        .5, np.ones(1), 1., observer.communication)
        self.assertFalse(accepted)
        self.assertEqual(observer.calls, calls)

    def test_nonfinite_site_event_never_completes_diagnostic(self):
        sites = [(np.ones((4, 1)), np.array([0., 0., 1., 1.]))]
        cfg = DslassoConfig(max_iter=1000)
        observer = kernel_observer(sites, cfg, RecoveryPolicy())
        observer.communication.incomplete_evaluations.append(dict(call=0, completed_sites=0,
            failed_site_index=0, reason="nonfinite_predictor", width=1, total_sites=1))
        with self.assertRaises(OptimizationFailure) as caught:
            proximal_fit(sites, np.zeros(1), 1, 0., cfg, observer.communication, observer)
        self.assertEqual(caught.exception.reason, "nonfinite_evaluation")

    def test_full_failure_retains_timing_context_and_prior_folds(self):
        cfg = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1000)
        original = proximal_fit

        def fail_later(sites, initial, penalty_start, penalty, config, communication=None, recovery=None):
            if recovery is not None and recovery.fit_context["fold"] == 1:
                # A genuine budget exit at the prescribed limit for this fixture:
                # force the start away from its exact intercept solution.
                initial = initial.copy()
                initial[0] += 1.
                config = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1, tolerance=1e-12)
            return original(sites, initial, penalty_start, penalty, config, communication, recovery)

        with patch("effect_modifier_experiment.survival.dslasso.proximal_fit", side_effect=fail_later):
            with self.assertRaises(OptimizationFailure) as caught:
                fit_recovery_dslasso(tiny_sites(), 3., config=cfg,
                    context={"scenario": "iid", "seed": 9, "recipe": "baseline"})
        d = caught.exception.diagnostics
        self.assertEqual(d["failure"]["context"]["fold"], 1)
        self.assertEqual(d["failure"]["context"]["seed"], 9)
        self.assertEqual(len(d["paths"][0]["fits"]), 2)
        self.assertGreater(d["timing_seconds"]["cross_validation"], 0)
        self.assertIsNone(d["timing_seconds"]["refit"])
        self.assertEqual(d["stage_status"]["refit"], "not_started")
        self.assertEqual(d["communication"]["phases"]["validation_path_loss"]["rounds"], 1)
        json.dumps(d, allow_nan=False)

    def test_complete_iteration_limit_validator_and_tampering(self):
        cfg = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1, tolerance=1e-12)
        with self.assertRaises(OptimizationFailure) as caught:
            fit_recovery_dslasso(tiny_sites(True), 3., config=cfg)
        error = caught.exception
        validate_recovery_diagnostics(error.diagnostics, cfg, RecoveryPolicy(), error.arrays, [32, 32], False)
        for change in ("calls", "timing", "iteration", "policy"):
            bad = copy.deepcopy(error.diagnostics)
            if change == "calls":
                bad["loss_gradient_calls"] += 1
            elif change == "timing":
                bad["timing_seconds"]["cross_validation"] = None
            elif change == "iteration":
                bad["paths"][-1]["unpenalized_start"]["iterations"] = 0
            else:
                bad["recovery_policy"]["max_step"] = 64.
            with self.assertRaises(ValueError):
                validate_recovery_diagnostics(bad, cfg, RecoveryPolicy(), error.arrays, [32, 32])

    def test_saved_full_evidence_rejects_fold_trace_payload_and_timing_forgery(self):
        cfg = DslassoConfig(num_folds=2, num_lambdas=2, max_iter=1000)
        model, diagnostics = fit_recovery_dslasso(tiny_sites(), 3., config=cfg)
        arrays = dict(intercept=np.array(model.intercept), coefficients=model.coefficients)
        def check(edit):
            bad = copy.deepcopy(diagnostics)
            edit(bad)
            with self.assertRaises(ValueError):
                validate_recovery_diagnostics(bad, cfg, RecoveryPolicy(), arrays, [32, 32])
        check(lambda d: d["site_folds"][0]["validation_sizes"].__setitem__(0, 17))
        check(lambda d: d["paths"][0]["partition_site_sizes"].__setitem__(0, 17))
        check(lambda d: d["validation_progress"].pop())
        check(lambda d: d["paths"][0]["unpenalized_start"]["trace"][0].__setitem__("smooth_objective", .9))
        check(lambda d: d["communication"]["phases"]["loss_gradient"].__setitem__("upload_bytes", 0))
        check(lambda d: d["stage_status"].__setitem__("refit", "failed"))
        check(lambda d: d["paths"][0]["unpenalized_start"].__setitem__("elapsed_wall_seconds", 1e9))


if __name__ == "__main__":
    unittest.main()
