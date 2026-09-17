import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from effect_modifier_experiment.survival.dslasso import SurvivalLinearModel
from effect_modifier_experiment.survival.evaluation import (
    evaluate_model, signal_recovery, treatment_contrast, validate_evaluation,
)


class SurvivalEvaluationTests(unittest.TestCase):
    def test_all_zero_scores_do_not_get_perfect_top_two(self):
        mask = np.arange(100) < 2
        metrics = signal_recovery(np.zeros(100), mask)
        self.assertAlmostEqual(metrics["probability_all_signals_selected"], 1 / math.comb(100, 2))
        self.assertEqual(metrics["auroc"], .5)

    def test_true_signals_and_boundary_tie(self):
        mask = np.array([True, True, False, False])
        perfect = signal_recovery([3, 2, 0, 0], mask)
        self.assertEqual(perfect["probability_all_signals_selected"], 1)
        tied = signal_recovery([3, 2, 2, 2], mask)
        self.assertAlmostEqual(tied["probability_all_signals_selected"], 1 / 3)

    def test_prognostic_effects_cancel_on_link_scale(self):
        x = np.array([[-1., 0., 2., 1.], [1., 2., -2., 0.], [0., -1., 1., 2.]])
        coefficient = np.array([.3, 1., 1., .5, -.5, 2., -2., 0., 0.])
        model = SurvivalLinearModel(-.4, coefficient)
        expected = .3 + 2 * x[:, 0] - 2 * x[:, 1]
        np.testing.assert_allclose(treatment_contrast(model.predict_eta, x), expected)
        dataset = SimpleNamespace(
            evaluation=SimpleNamespace(x=x),
            evaluation_truth=SimpleNamespace(delta=expected,
                                             modifier_mask=np.array([True, True, False, False])))
        result = evaluate_model(model.predict_eta, dataset, repeats=3, seed=0)
        self.assertLess(result["contrast_rmse"], 1e-14)
        self.assertLess(max(result["importance"][2:]), 1e-14)
        validate_evaluation(result, 4, 3, 3)
        altered = copy.deepcopy(result)
        altered["recovery"]["auroc"] = .5
        with self.assertRaises(ValueError):
            validate_evaluation(altered, 4, 3, 3)


if __name__ == "__main__":
    unittest.main()
