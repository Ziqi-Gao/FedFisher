import dataclasses
import inspect
import io
from contextlib import redirect_stdout
import unittest

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from data import get_dataset
from effect_modifier_experiment.one_shot_lasso_benchmark import (
    LinearSitePayload,
    LocalLogisticConfig,
    MlpGroupLassoConfig,
    MlpSitePayload,
    SealedTrainingSite,
    aggregate_linear_payloads,
    aggregate_mlp_payloads,
    apply_interaction_group_prox,
    evaluation_communication_report,
    fit_local_logistic_lasso_site,
    fit_mlp_group_lasso_site,
    make_common_mlp_state,
    payload_fingerprint,
    tensor_datasets_to_sealed_sites,
    training_fingerprint,
)
from effect_modifier_experiment.one_shot_lasso_run_contract import (
    METHOD_NAMES,
    PRIVACY_LABEL,
    canonical_json,
    config_hashes,
    validate_manifest,
    validate_run_id,
)
from effect_modifier_experiment.run_one_shot_lasso_effect_modifier import (
    parse_args,
)
from models import get_model


class OneShotLassoCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    def _datasets(self, gaussian=False, split="iid", covariate_dim=4, seed=11):
        with redirect_stdout(io.StringIO()):
            sites, _, test, _ = get_dataset(
                datatype="SyntheticEffectModifier",
                n_client=10,
                n_c=2,
                alpha=0.1,
                synthetic_split=split,
                synthetic_num_test=80,
                effect_modifier_covariate_dim=covariate_dim,
                effect_modifier_signal_dim=2,
                effect_modifier_signal_strength=2.0,
                effect_modifier_main_effect_strength=1.0,
                effect_modifier_confounding_strength=1.0,
                effect_modifier_intercept=0.0,
                effect_modifier_treatment_prob=0.5,
                effect_modifier_propensity_clip=0.05,
                effect_modifier_site_min_samples=80,
                effect_modifier_site_max_samples=90,
                effect_modifier_site_gaussian_shift=gaussian,
                seed=seed,
            )
        return sites, test

    def test_linear_payload_weighting_round_and_size(self):
        payloads = [
            LinearSitePayload(0, 1, np.arange(6, dtype=np.float64)),
            LinearSitePayload(1, 3, np.arange(6, dtype=np.float64) + 4.0),
        ]
        result, report = aggregate_linear_payloads(payloads, covariate_dim=2)
        expected = 0.25 * payloads[0].coefficients + 0.75 * payloads[1].coefficients
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(report["training_communication_rounds"], 1)
        self.assertEqual(report["site_upload_count"], 2)
        self.assertEqual(report["post_upload_training_site_accesses"], 0)
        self.assertEqual(report["payload_vector_elements_per_site"], 6)
        self.assertEqual(report["payload_bytes_per_site"], 64)

    def test_payload_schema_forbids_gradient_hessian_fisher_and_callback(self):
        self.assertEqual(
            [field.name for field in dataclasses.fields(LinearSitePayload)],
            ["site_idx", "num_examples", "coefficients"],
        )
        self.assertEqual(
            [field.name for field in dataclasses.fields(MlpSitePayload)],
            ["site_idx", "num_examples", "model_vector"],
        )
        forbidden = {"gradient", "hessian", "fisher", "dataset", "site_callback"}
        for payload_type in (LinearSitePayload, MlpSitePayload):
            self.assertTrue(forbidden.isdisjoint({field.name for field in dataclasses.fields(payload_type)}))
        self.assertNotIn("site", inspect.signature(aggregate_linear_payloads).parameters)
        self.assertNotIn("site", inspect.signature(aggregate_mlp_payloads).parameters)

    def test_local_logistic_fit_seals_site_and_uploads_exact_dimension(self):
        sites, _ = self._datasets(covariate_dim=2)
        sealed = tensor_datasets_to_sealed_sites([sites[0]], 2)[0]
        payload, audit = fit_local_logistic_lasso_site(
            sealed,
            LocalLogisticConfig(
                covariate_dim=2,
                cv_folds=2,
                lambda_count=2,
                lambda_min_ratio=0.2,
                optimizer_max_iter=2000,
                optimizer_tolerance=1e-5,
            ),
            experiment_seed=11,
        )
        self.assertEqual(payload.coefficients.shape, (6,))
        self.assertEqual(payload.coefficients.dtype, np.float64)
        self.assertFalse(payload.coefficients.flags.writeable)
        self.assertEqual(audit["pre_upload_training_communication_rounds"], 0)
        self.assertTrue(sealed.sealed)
        with self.assertRaisesRegex(RuntimeError, "after payload upload"):
            sealed.arrays()

    def test_group_prox_changes_only_interaction_first_layer_columns(self):
        torch.manual_seed(4)
        model = get_model("SyntheticMLP", 2, bias=False, synthetic_dim=7)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        report = apply_interaction_group_prox(model, covariate_dim=3, threshold=100.0)
        after = model.state_dict()
        self.assertEqual(report["penalty_start_input"], 4)
        self.assertEqual(report["penalty_end_input_exclusive"], 7)
        self.assertTrue(torch.equal(after["net.0.weight"][:, :4], before["net.0.weight"][:, :4]))
        self.assertEqual(torch.count_nonzero(after["net.0.weight"][:, 4:]).item(), 0)
        for key in before:
            if key != "net.0.weight":
                self.assertTrue(torch.equal(after[key], before[key]), key)

    def test_mlp_site_local_selection_upload_and_seal(self):
        sites, _ = self._datasets(covariate_dim=2)
        sealed = tensor_datasets_to_sealed_sites([sites[0]], 2)[0]
        state = make_common_mlp_state("SyntheticMLP", 5, seed=11)
        payload, audit = fit_mlp_group_lasso_site(
            sealed,
            "SyntheticMLP",
            state,
            MlpGroupLassoConfig(
                covariate_dim=2,
                local_epochs=1,
                batch_size=8,
                lambda_ratios=(0.0, 1.0),
            ),
            experiment_seed=11,
        )
        template = get_model("SyntheticMLP", 2, bias=False, synthetic_dim=5)
        self.assertEqual(payload.model_vector.size, parameters_to_vector(template.parameters()).numel())
        self.assertEqual(payload.model_vector.dtype, np.float32)
        self.assertFalse(payload.model_vector.flags.writeable)
        self.assertEqual(len(audit["lambda_candidates"]), 2)
        self.assertIn(audit["selected_ratio"], (0.0, 1.0))
        self.assertTrue(sealed.sealed)
        with self.assertRaises(RuntimeError):
            sealed.arrays()

    def test_mlp_aggregation_is_exact_existing_fedavg(self):
        state = make_common_mlp_state("SyntheticMLP", 5, seed=2)
        template = get_model("SyntheticMLP", 2, bias=False, synthetic_dim=5)
        template.load_state_dict(state)
        base = parameters_to_vector(template.parameters()).detach().numpy()
        payloads = [
            MlpSitePayload(0, 2, base + np.float32(1.0)),
            MlpSitePayload(1, 6, base - np.float32(1.0)),
        ]
        model, report = aggregate_mlp_payloads(payloads, "SyntheticMLP", 5, state)
        expected = 0.25 * payloads[0].model_vector + 0.75 * payloads[1].model_vector
        np.testing.assert_allclose(
            parameters_to_vector(model.parameters()).detach().numpy(),
            expected,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(report["training_communication_rounds"], 1)
        self.assertIn("get_fedavg_model", report["server_aggregation"])

    def test_factorial_training_and_test_fingerprints(self):
        control_sites, control_test = self._datasets(False, "noniid")
        local_sites, local_test = self._datasets(False, "noniid")
        gaussian_sites, gaussian_test = self._datasets(True, "noniid")
        combined_sites, combined_test = self._datasets(True, "noniid")
        control = tensor_datasets_to_sealed_sites(control_sites, 4)
        local = tensor_datasets_to_sealed_sites(local_sites, 4)
        gaussian = tensor_datasets_to_sealed_sites(gaussian_sites, 4)
        combined = tensor_datasets_to_sealed_sites(combined_sites, 4)
        self.assertEqual(training_fingerprint(control), training_fingerprint(local))
        self.assertEqual(training_fingerprint(gaussian), training_fingerprint(combined))
        from effect_modifier_experiment.dslassocov_benchmark import dataset_fingerprint

        self.assertEqual(dataset_fingerprint(control_test), dataset_fingerprint(local_test))
        self.assertEqual(dataset_fingerprint(gaussian_test), dataset_fingerprint(combined_test))
        self.assertEqual(dataset_fingerprint(control_test), dataset_fingerprint(gaussian_test))

    def test_payload_fingerprint_is_ordered_and_sensitive(self):
        first = [
            LinearSitePayload(0, 2, np.zeros(6)),
            LinearSitePayload(1, 3, np.ones(6)),
        ]
        second = [
            LinearSitePayload(0, 2, np.zeros(6)),
            LinearSitePayload(1, 3, np.ones(6) * 2),
        ]
        self.assertNotEqual(payload_fingerprint(first), payload_fingerprint(second))

    def test_evaluation_communication_is_separate(self):
        self.assertEqual(
            evaluation_communication_report("global", 10)["evaluation_communication_rounds"],
            0,
        )
        site = evaluation_communication_report("site", 10)
        self.assertEqual(site["evaluation_communication_rounds"], 1)
        self.assertEqual(site["site_score_uploads"], 10)
        self.assertEqual(site["training_site_accesses"], 0)

    def test_manifest_recomputes_hashes_and_variant_contract(self):
        family = "one_shot_local_logistic_lasso_fedavg"
        run_config = {"a": 1}
        method_config = {"b": 2}
        variant_config = {
            "effect_modifier_site_gaussian_shift": False,
            "prediction_intervention_scope": "global",
        }
        hashes = config_hashes(run_config, method_config, variant_config)
        manifest = {
            "format_version": "2",
            "run_id": family + "_control_test",
            "experiment_family": family,
            "method_name": METHOD_NAMES[family],
            "privacy_label": PRIVACY_LABEL,
            "effect_modifier_variant": "control",
            "training_communication_rounds": "1",
            "post_upload_training_site_access_allowed": "0",
            "run_config_json": canonical_json(run_config),
            "method_config_json": canonical_json(method_config),
            "variant_config_json": canonical_json(variant_config),
            **hashes,
        }
        validate_manifest(manifest)
        manifest["method_config_json"] = canonical_json({"b": 3})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_manifest(manifest)

    def test_run_id_and_family_model_validation(self):
        validate_run_id(
            "one_shot_local_logistic_lasso_fedavg_control_smoke",
            "one_shot_local_logistic_lasso_fedavg",
            "control",
        )
        with self.assertRaises(ValueError):
            validate_run_id(
                "dslassocov_control_smoke",
                "one_shot_local_logistic_lasso_fedavg",
                "control",
            )
        with self.assertRaisesRegex(ValueError, "requires --model linear"):
            parse_args(
                [
                    "--family",
                    "one_shot_local_logistic_lasso_fedavg",
                    "--variant",
                    "control",
                    "--run-id",
                    "one_shot_local_logistic_lasso_fedavg_control_test",
                    "--output-dir",
                    "/data/del6500/FedFisher/effect_modifier_experiment/outputs/one_shot_local_logistic_lasso_fedavg_control_test",
                    "--split",
                    "iid",
                    "--model",
                    "SyntheticMLP",
                ]
            )


if __name__ == "__main__":
    unittest.main()
