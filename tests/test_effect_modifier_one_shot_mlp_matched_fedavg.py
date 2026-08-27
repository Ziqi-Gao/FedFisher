import unittest

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from effect_modifier_experiment.one_shot_lasso_benchmark import (
    MlpGroupLassoConfig,
    SealedTrainingSite,
    aggregate_mlp_payloads,
    fit_mlp_group_lasso_site,
    make_common_mlp_state,
)
from effect_modifier_experiment.one_shot_mlp_matched_fedavg import (
    final_refit_loader_seed,
    fit_matched_zero_penalty_site,
)


class OneShotMlpMatchedFedAvgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    def test_lambda_zero_is_exact_matched_one_shot_fedavg(self):
        rng = np.random.default_rng(83041)
        covariate_dim = 2
        num_examples = 40
        treatment = rng.binomial(1, 0.5, size=num_examples).astype(np.float64)
        raw_x = rng.normal(size=(num_examples, covariate_dim))
        features = np.concatenate(
            (treatment[:, None], raw_x, treatment[:, None] * raw_x), axis=1
        )
        labels = np.arange(num_examples, dtype=np.int64) % 2
        config = MlpGroupLassoConfig(
            covariate_dim=covariate_dim,
            local_epochs=2,
            batch_size=8,
            lambda_ratios=(0.0,),
        )

        for model_name in ("SyntheticMLP", "SyntheticMLPDeep"):
            with self.subTest(model=model_name):
                common_state = make_common_mlp_state(
                    model_name, 1 + 2 * covariate_dim, seed=13
                )
                penalized_site = SealedTrainingSite(0, features, labels)
                matched_site = SealedTrainingSite(0, features, labels)
                zero_payload, zero_audit = fit_mlp_group_lasso_site(
                    penalized_site,
                    model_name,
                    common_state,
                    config,
                    experiment_seed=13,
                )
                matched_payload, matched_audit = fit_matched_zero_penalty_site(
                    matched_site,
                    model_name,
                    common_state,
                    config,
                    experiment_seed=13,
                )

                self.assertEqual(zero_audit["selected_lambda"], 0.0)
                self.assertEqual(zero_audit["selected_ratio"], 0.0)
                self.assertEqual(
                    zero_audit["final_refit_loader_seed"],
                    final_refit_loader_seed(config, 13, 0),
                )
                self.assertEqual(
                    zero_audit["final_refit_loader_seed"], matched_audit["loader_seed"]
                )
                np.testing.assert_array_equal(
                    zero_payload.model_vector, matched_payload.model_vector
                )

                zero_model, _ = aggregate_mlp_payloads(
                    [zero_payload], model_name, 1 + 2 * covariate_dim, common_state
                )
                matched_model, _ = aggregate_mlp_payloads(
                    [matched_payload], model_name, 1 + 2 * covariate_dim, common_state
                )
                np.testing.assert_array_equal(
                    parameters_to_vector(zero_model.parameters()).detach().numpy(),
                    parameters_to_vector(matched_model.parameters()).detach().numpy(),
                )
                self.assertTrue(penalized_site.sealed)
                self.assertTrue(matched_site.sealed)


if __name__ == "__main__":
    unittest.main()
