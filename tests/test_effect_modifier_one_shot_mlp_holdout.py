import unittest

import numpy as np
import torch

from effect_modifier_experiment.one_shot_lasso_benchmark import (
    MlpGroupLassoConfig,
    SealedTrainingSite,
    fit_mlp_group_lasso_site,
    make_common_mlp_state,
)


class OneShotMlpObservedClassHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    def test_pure_label_site_uses_observed_class_holdout_and_uploads_once(self):
        rng = np.random.default_rng(82031)
        num_examples = 40
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
        site = SealedTrainingSite(
            0, features, np.ones(num_examples, dtype=np.int64)
        )
        common_state = make_common_mlp_state(
            "SyntheticMLP", 1 + 2 * covariate_dim, seed=0
        )
        payload, audit = fit_mlp_group_lasso_site(
            site,
            "SyntheticMLP",
            common_state,
            MlpGroupLassoConfig(
                covariate_dim=covariate_dim,
                local_epochs=1,
                batch_size=8,
                lambda_ratios=(0.0, 1.0),
            ),
            experiment_seed=0,
        )

        self.assertTrue(site.sealed)
        self.assertTrue(np.isfinite(payload.model_vector).all())
        self.assertEqual(
            audit["lambda_selection_scope"],
            "deterministic_site_local_observed_class_stratified_holdout",
        )
        self.assertEqual(audit["holdout_original_label_counts"], {"0": 0, "1": 40})
        self.assertEqual(audit["holdout_training_examples"], 32)
        self.assertEqual(audit["holdout_validation_examples"], 8)
        self.assertEqual(audit["pre_upload_training_communication_rounds"], 0)
        with self.assertRaisesRegex(RuntimeError, "after payload upload"):
            site.arrays()


if __name__ == "__main__":
    unittest.main()
