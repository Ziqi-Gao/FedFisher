import unittest

import numpy as np

from effect_modifier_experiment.one_shot_lasso_benchmark import (
    LocalLogisticConfig,
    SealedTrainingSite,
    fit_local_logistic_lasso_site,
)
from effect_modifier_experiment.one_shot_local_logistic import SMOOTHING_RULE


class OneShotLocalLogisticSeparationTests(unittest.TestCase):
    def test_pure_label_site_has_finite_audited_fit_and_is_sealed(self):
        rng = np.random.default_rng(81021)
        num_examples = 80
        covariate_dim = 2
        treatment = rng.binomial(1, 0.5, size=num_examples).astype(np.float64)
        raw_x = rng.normal(size=(num_examples, covariate_dim))
        features = np.concatenate(
            (
                treatment[:, None],
                raw_x,
                treatment[:, None] * raw_x,
            ),
            axis=1,
        )
        labels = np.ones(num_examples, dtype=np.int64)
        site = SealedTrainingSite(0, features, labels)
        payload, audit = fit_local_logistic_lasso_site(
            site,
            LocalLogisticConfig(
                covariate_dim=covariate_dim,
                cv_folds=2,
                lambda_count=3,
                lambda_min_ratio=0.1,
                optimizer_max_iter=5000,
                optimizer_tolerance=1e-5,
            ),
            experiment_seed=0,
        )

        self.assertTrue(site.sealed)
        self.assertEqual(payload.coefficients.shape, (2 + 2 * covariate_dim,))
        self.assertTrue(np.isfinite(payload.coefficients).all())
        stabilization = audit["response_stabilization"]
        self.assertEqual(stabilization["rule"], SMOOTHING_RULE)
        self.assertAlmostEqual(stabilization["epsilon"], 1.0 / 81.0)
        self.assertEqual(stabilization["original_label_counts"], {"0": 0, "1": 80})
        self.assertFalse(
            stabilization["coefficient_penalty_added_to_unpenalized_block"]
        )
        self.assertTrue(audit["selected_final_report"]["converged"])
        self.assertTrue(
            all(report["converged"] for report in audit["selected_cv_reports"])
        )


if __name__ == "__main__":
    unittest.main()
