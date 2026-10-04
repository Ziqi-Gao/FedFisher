"""Exact small arrays verify probability metrics independently of training."""

import copy
from types import SimpleNamespace
import unittest

import numpy as np

from effect_modifier_experiment.survival.links import survival_link, survival_mean
from effect_modifier_experiment.survival_pooled.evaluation import (
    compare_predictions, probability_predictions, summarize_arrays,
    summarize_predictions, validate_comparison, validate_summary_arrays,
)


def truth_arrays():
    eta0 = survival_link(np.array([.2, .4, .6, .8]))
    eta1 = survival_link(np.array([.1, .3, .7, .9]))
    return dict(eta0=eta0, eta1=eta1, survival0=survival_mean(eta0), survival1=survival_mean(eta1))


def predictions(s0, s1):
    eta0, eta1 = survival_link(s0), survival_link(s1)
    return dict(eta0=eta0, eta1=eta1, s0=survival_mean(eta0), s1=survival_mean(eta1))


class SurvivalPooledEvaluationTests(unittest.TestCase):
    def test_oracle_predictions_have_zero_error_in_every_arm_and_factual(self):
        truth = truth_arrays()
        predicted = dict(eta0=truth["eta0"], eta1=truth["eta1"],
                         s0=truth["survival0"], s1=truth["survival1"])
        treatment = np.array([0, 1, 0, 1])
        result = summarize_arrays(predicted, truth, treatment)
        for metrics in list(result["arms"].values()) + [result["combined"], result["factual"], result["contrast"]]:
            self.assertEqual(metrics["rmse"], 0)
            self.assertEqual(metrics["mae"], 0)
            self.assertEqual(metrics["bias"], 0)
        self.assertEqual(result["combined"]["n"], 8)
        for arm in ("0", "1"):
            self.assertEqual(len(result["calibration"][arm]), 10)
            self.assertEqual(sum(row["count"] for row in result["calibration"][arm]), 4)
        validate_summary_arrays(result, predicted, truth, treatment)
        altered = copy.deepcopy(result)
        altered["factual"]["rmse"] = .3
        with self.assertRaises(ValueError):
            validate_summary_arrays(altered, predicted, truth, treatment)

    def test_bias_rmse_mae_and_factual_selection_match_direct_arithmetic(self):
        truth = truth_arrays()
        predicted = predictions(np.array([.3, .4, .5, .8]), np.array([.1, .4, .9, .7]))
        treatment = np.array([0, 1, 0, 1])
        result = summarize_arrays(predicted, truth, treatment)
        e0 = predicted["s0"] - truth["survival0"]
        e1 = predicted["s1"] - truth["survival1"]
        factual = np.where(treatment == 1, e1, e0)
        self.assertAlmostEqual(result["arms"]["0"]["rmse"], np.sqrt(np.mean(e0 ** 2)))
        self.assertAlmostEqual(result["arms"]["1"]["mae"], np.mean(np.abs(e1)))
        self.assertAlmostEqual(result["combined"]["mse"], np.mean(np.r_[e0, e1] ** 2))
        self.assertAlmostEqual(result["factual"]["bias"], factual.mean())
        self.assertAlmostEqual(result["factual"]["rmse"], np.sqrt(np.mean(factual ** 2)))

    def test_counterfactual_design_rebuilds_treatment_and_interactions(self):
        x = np.array([[-1., 2.], [0., 3.], [2., -1.], [.2, -.3]])
        truth = truth_arrays()
        dataset = SimpleNamespace(evaluation=SimpleNamespace(x=x, a=np.array([0, 1, 0, 1])),
                                  evaluation_truth=SimpleNamespace(**truth))
        calls = []

        def predictor(design):
            calls.append(design.copy())
            return design[:, 0] + design[:, 1] + 2 * design[:, 3]

        predicted = probability_predictions(predictor, dataset, batch_size=2)
        np.testing.assert_array_equal(predicted["eta0"], x[:, 0])
        np.testing.assert_array_equal(predicted["eta1"], 1 + 3 * x[:, 0])
        for call in calls:
            np.testing.assert_array_equal(call[:, 3:], call[:, :1] * call[:, 1:3])
        self.assertEqual(summarize_predictions(predicted, dataset), summarize_arrays(predicted, truth, dataset.evaluation.a))

    def test_extreme_survival_values_belong_to_first_and_last_bins(self):
        eta = np.array([-1000., 1000.])
        predicted = dict(eta0=eta, eta1=eta, s0=survival_mean(eta), s1=survival_mean(eta))
        truth = dict(eta0=eta, eta1=eta, survival0=survival_mean(eta), survival1=survival_mean(eta))
        result = summarize_arrays(predicted, truth, np.array([0, 1]))
        self.assertEqual(result["calibration"]["0"][0]["count"], 1)
        self.assertEqual(result["calibration"]["0"][9]["count"], 1)
        self.assertIsNone(result["calibration"]["0"][1]["predicted_mean"])

    def test_paired_model_gap_and_invalid_link_arrays(self):
        left = predictions(np.array([.2, .5]), np.array([.3, .8]))
        right = predictions(np.array([.3, .4]), np.array([.5, .6]))
        gap = compare_predictions(left, right)
        self.assertAlmostEqual(gap["arms"]["0"]["rmse"], .1)
        self.assertAlmostEqual(gap["arms"]["1"]["rmse"], .2)
        self.assertAlmostEqual(gap["combined"]["rmse"], np.sqrt(.025))
        validate_comparison(gap, left, right)
        altered = copy.deepcopy(left)
        altered["s0"][0] += .01
        with self.assertRaises(ValueError):
            compare_predictions(altered, right)


if __name__ == "__main__":
    unittest.main()
