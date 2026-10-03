"""Analytic and finite-difference fixtures; no scientific fit or pilot."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from effect_modifier_experiment.survival.data import SurvivalRows
from effect_modifier_experiment.survival.links import (
    pseudo_mse_loss_gradient, survival_link, survival_mean, survival_mean_derivative,
)
from effect_modifier_experiment.survival.pseudo import (
    km_survival, pseudo_survival, site_pseudo_response, site_pseudo_diagnostics,
)


def weighted_product_limit(time, event, weights, tau):
    """Independent slow KM definition for numerical differentiation only."""
    answer = 1.0
    for cut in sorted(set(time[np.asarray(event, dtype=bool) & (time <= tau)])):
        risk = sum(weights[i] for i in range(len(time)) if time[i] >= cut)
        deaths = sum(weights[i] for i in range(len(time)) if time[i] == cut and event[i])
        answer *= 1.0 - deaths / risk
    return answer


class SurvivalPseudoTests(unittest.TestCase):
    def test_uncensored_values_reduce_to_survival_indicator(self):
        time = np.array([4.0, 1.0, 2.0, 2.0, 3.0])
        for tau in (0.0, 1.0, 2.0, 3.5, 4.0):
            np.testing.assert_allclose(pseudo_survival(time, np.ones(5), tau), time > tau,
                                       rtol=0, atol=8e-16)

    def test_censoring_can_produce_negative_and_greater_than_one_values(self):
        # At t=2, risk=2 and deaths=1: S=1/2, dS/dw=(0,-1/4,+1/4).
        expected = np.array([0.5, -0.25, 1.25])
        np.testing.assert_allclose(pseudo_survival([1, 2, 3], [0, 1, 1], 2), expected)
        self.assertEqual(km_survival([1, 2, 3], [0, 1, 1], 2), 0.5)
        self.assertEqual(expected.mean(), 0.5)

    def test_infinitesimal_jackknife_agrees_with_weight_perturbations(self):
        fixtures = [([1, 2, 3, 4, 5], [0, 1, 1, 0, 1], 3),
                    ([1, 1, 2, 2, 2, 3, 4], [1, 0, 1, 1, 0, 0, 1], 2),
                    ([0, 0, 0.2, 1, 2], [1, 0, 1, 0, 1], 0.3)]
        for raw_time, raw_event, tau in fixtures:
            time, event = np.array(raw_time), np.array(raw_event)
            n, epsilon = len(time), 1e-6
            base = weighted_product_limit(time, event, np.ones(n), tau)
            expected = np.empty(n)
            for i in range(n):
                plus, minus = np.ones(n), np.ones(n)
                plus[i] += epsilon
                minus[i] -= epsilon
                derivative = (weighted_product_limit(time, event, plus, tau)
                              - weighted_product_limit(time, event, minus, tau)) / (2 * epsilon)
                expected[i] = base + n * derivative
            actual = pseudo_survival(time, event, tau, timefix=False)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
            self.assertAlmostEqual(float(actual.mean()), base, places=14)

    def test_not_leave_one_out_jackknife(self):
        time, event, tau = np.array([1, 2, 3]), np.array([0, 1, 1]), 2
        exact_jackknife = np.array([0.5, -0.5, 1.5])
        values = pseudo_survival(time, event, tau)
        self.assertFalse(np.allclose(values, exact_jackknife))

    def test_tied_censor_is_at_risk_for_event_at_same_time(self):
        # Both people observed at time 1 count in that event's risk set.
        np.testing.assert_allclose(pseudo_survival([1, 1, 2], [1, 0, 0], 1), [0, 1, 1], atol=1e-15)
        self.assertAlmostEqual(km_survival([1, 1, 2], [1, 0, 0], 1), 2 / 3)

    def test_timefix_matches_r_aeqsurv_near_tie_rule(self):
        time, event = [1, 1 + 1e-9, 2], [0, 1, 0]
        np.testing.assert_allclose(pseudo_survival(time, event, 1.5), [1, 0, 1], atol=1e-15)
        np.testing.assert_allclose(pseudo_survival(time, event, 1.5, timefix=False), [0.5, -0.25, 1.25])
        # Relative tolerance is based on mean absolute unique time.
        np.testing.assert_allclose(pseudo_survival(np.array(time) * 1e4, event, 1.5e4),
                                   [1, 0, 1], atol=1e-15)

    def test_endpoints_and_beyond_curve_warning(self):
        np.testing.assert_array_equal(pseudo_survival([1, 2], [0, 0], 2), [1, 1])
        np.testing.assert_array_equal(pseudo_survival([1, 2, 2], [0, 1, 1], 2), [0, 0, 0])
        np.testing.assert_array_equal(pseudo_survival([0], [1], 0), [0])
        with self.assertWarns(RuntimeWarning):
            np.testing.assert_array_equal(pseudo_survival([1, 2], [1, 0], 3), [0, 1])

    def test_permutation_equivariance_and_locality(self):
        rows = SurvivalRows(np.zeros((4, 2)), [0, 1, 0, 1], [1, 2, 3, 4], [0, 1, 0, 1], np.arange(4))
        original = site_pseudo_response(rows, 2.0)
        order = np.array([2, 0, 3, 1])
        np.testing.assert_array_equal(site_pseudo_response(rows.subset(order), 2), original[order])
        subset = rows.subset([0, 1, 2])
        np.testing.assert_allclose(site_pseudo_response(subset, 2), [0.5, -0.25, 1.25])
        self.assertFalse(np.allclose(site_pseudo_response(subset, 2), original[:3]))

    def test_invalid_observations_fail_closed(self):
        cases = [([], [], 1), ([1], [2], 1), ([1, 2], [1], 1),
                 ([-1, 2], [0, 1], 1), ([np.inf], [0], 1), ([1], [0], np.nan),
                 ([1], [0], -1), ([1], [0], True), ([1], [0], [1])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                pseudo_survival(*args)

    def test_observable_diagnostics_and_cached_pseudo_values(self):
        rows = SurvivalRows(np.zeros((3, 2)), [0, 1, 0], [1, 2, 3], [0, 1, 1], np.arange(3))
        expected = dict(n=3, event_count=2, censored_count=1, censoring_fraction=1 / 3,
                        n_at_risk_tau=2, km_survival_tau=0.5, pv_min=-0.25, pv_max=1.25,
                        pv_mean=0.5, outside_unit_interval_count=2, all_zero=False, all_one=False,
                        tau=2.0, max_observed_time=3.0, tau_beyond_observed_max=False)
        self.assertEqual(site_pseudo_diagnostics(rows, 2), expected)
        cached = site_pseudo_response(rows, 2)
        with patch("effect_modifier_experiment.survival.pseudo.site_pseudo_response",
                   side_effect=AssertionError("cached PV must not be reconstructed")):
            self.assertEqual(site_pseudo_diagnostics(rows, 2, cached), expected)
        self.assertEqual(json.loads(json.dumps(expected, allow_nan=False)), expected)
        # The reported fraction is 1/3; the DGP's exponential rate 0.05
        # does not enter these observed-data diagnostics at all.
        self.assertNotEqual(expected["censoring_fraction"], 0.05)
        for invalid in ([0, 1], [0, 1, np.inf], [[0, 1, 1]], [1e308, 1e308, 1e308]):
            with self.assertRaises(ValueError):
                site_pseudo_diagnostics(rows, 2, invalid)

    def test_diagnostic_degeneracy_and_literal_tau_support(self):
        censored = SurvivalRows(np.zeros((2, 2)), [0, 1], [1, 2], [0, 0], np.arange(2))
        at_maximum = site_pseudo_diagnostics(censored, 2)
        self.assertTrue(at_maximum["all_one"])
        self.assertFalse(at_maximum["all_zero"])
        self.assertFalse(at_maximum["tau_beyond_observed_max"])
        self.assertEqual(at_maximum["n_at_risk_tau"], 1)
        with self.assertWarns(RuntimeWarning):
            beyond = site_pseudo_diagnostics(censored, 2 + 1e-10, np.ones(2))
        self.assertTrue(beyond["tau_beyond_observed_max"])
        self.assertEqual(beyond["n_at_risk_tau"], 0)
        failed = SurvivalRows(np.zeros((2, 2)), [0, 1], [1, 2], [1, 1], np.arange(2))
        no_survivors = site_pseudo_diagnostics(failed, 2)
        self.assertTrue(no_survivors["all_zero"])
        self.assertFalse(no_survivors["all_one"])
        self.assertEqual(no_survivors["censoring_fraction"], 0)


class SurvivalLinkTests(unittest.TestCase):
    def test_link_orientation_and_extreme_values(self):
        eta = np.array([-4.0, -1.0, 0, 1.0])
        np.testing.assert_allclose(survival_link(survival_mean(eta)), eta, atol=5e-15)
        self.assertAlmostEqual(float(survival_mean(0)), np.exp(-1))
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_array_equal(survival_mean([-np.inf, -1000, 1000, np.inf]), [1, 1, 0, 0])
            np.testing.assert_array_equal(survival_mean_derivative([-np.inf, -1000, 1000, np.inf]), [0, 0, 0, 0])
        for values in ([0], [1], [-0.2], [1.2], [np.nan]):
            with self.assertRaises(ValueError):
                survival_link(values)

    def test_response_loss_gradient_without_clipping_pseudo_values(self):
        eta, response = np.array([-1.0, 0.0, 0.5]), np.array([-0.25, 0.5, 1.25])
        loss, gradient = pseudo_mse_loss_gradient(eta, response)
        self.assertAlmostEqual(loss, 0.5 * np.mean((survival_mean(eta) - response) ** 2))
        epsilon = 1e-6
        for index in range(3):
            perturb = np.zeros(3)
            perturb[index] = epsilon
            numerical = (pseudo_mse_loss_gradient(eta + perturb, response)[0]
                         - pseudo_mse_loss_gradient(eta - perturb, response)[0]) / (2 * epsilon)
            self.assertAlmostEqual(gradient[index], numerical, places=9)


if __name__ == "__main__":
    unittest.main()
