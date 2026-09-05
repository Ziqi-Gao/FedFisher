import contextlib
import io
import unittest

import torch

from data import get_dataset
from effect_modifier_experiment.one_shot_lasso_benchmark import (
    DEFAULT_LOCAL_LOGISTIC_OPTIMIZER_MAX_ITER,
    LocalLogisticConfig,
    fit_local_logistic_lasso_site,
    tensor_datasets_to_sealed_sites,
)


class OneShotSeed285StationarityRegressionTests(unittest.TestCase):
    def test_gaussian_local_permutation_site_8_converges_without_relaxation(self):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        with contextlib.redirect_stdout(io.StringIO()):
            site_datasets, _, _, _ = get_dataset(
                datatype="SyntheticEffectModifier",
                n_client=10,
                n_c=2,
                alpha=0.1,
                synthetic_split="noniid",
                synthetic_num_test=10000,
                effect_modifier_covariate_dim=100,
                effect_modifier_signal_dim=2,
                effect_modifier_signal_strength=2.0,
                effect_modifier_main_effect_strength=1.0,
                effect_modifier_confounding_strength=1.0,
                effect_modifier_intercept=0.0,
                effect_modifier_treatment_prob=0.5,
                effect_modifier_propensity_clip=0.05,
                effect_modifier_site_min_samples=500,
                effect_modifier_site_max_samples=1000,
                effect_modifier_site_gaussian_shift=True,
                effect_modifier_site_gaussian_mean_span=0.75,
                effect_modifier_site_gaussian_std_low=0.75,
                effect_modifier_site_gaussian_std_high=1.25,
                seed=285,
            )
        site = tensor_datasets_to_sealed_sites(site_datasets, 100)[8]
        config = LocalLogisticConfig()
        self.assertEqual(
            config.optimizer_max_iter,
            DEFAULT_LOCAL_LOGISTIC_OPTIMIZER_MAX_ITER,
        )
        payload, audit = fit_local_logistic_lasso_site(site, config, 285)

        self.assertEqual(payload.site_idx, 8)
        self.assertEqual(audit["selected_index"], 23)
        self.assertEqual(
            audit["lambda_path"][audit["selected_index"]],
            audit["selected_lambda"],
        )
        reports = audit["selected_cv_reports"] + [audit["selected_final_report"]]
        self.assertTrue(all(report["converged"] for report in reports))
        self.assertGreater(max(report["iterations"] for report in reports), 2000)
        for report in reports:
            self.assertLessEqual(
                report["proximal_gradient_mapping_inf"],
                report["stationarity_threshold"],
            )
            self.assertLessEqual(
                report["kkt_residual_inf"],
                report["stationarity_threshold"],
            )
            self.assertFalse(report["momentum_reset_polish"]["attempted"])


if __name__ == "__main__":
    unittest.main()
