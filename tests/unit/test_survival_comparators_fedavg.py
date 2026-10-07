"""Bounded explicit observed fixtures for the transferred six-round recipe."""

import copy
from dataclasses import replace
import struct
import unittest
from unittest.mock import patch

import numpy as np
import torch

from effect_modifier_experiment.components.fingerprints import array_hash
from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite
from effect_modifier_experiment.survival.pseudo import site_pseudo_response
from effect_modifier_experiment.survival_mlp8.methods import public_initial_vector, _train
from effect_modifier_experiment.causal_comparison.fedavg import _loader_seed as prior_loader_seed
from effect_modifier_experiment.causal_comparison.fedavg import FedAvgConfig as PriorConfig
from effect_modifier_experiment.survival_comparators import fedavg


def observed_sites():
    rng = np.random.default_rng(381)
    return [SurvivalSite(index, SurvivalRows(rng.normal(size=(n, 2)), np.arange(n) % 2,
            np.resize([1., 2., 4., 5.], n), np.resize([1, 0, 1, 1], n),
            np.arange(index * 100, index * 100 + n))) for index, n in enumerate((7, 9))]


class SurvivalFedAvgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = fedavg.FedAvgConfig()
        cls.model, cls.report, cls.arrays = fedavg.fit_fedavg(observed_sites(), 3., 7, cls.config, -.47)

    def validate(self, report=None, arrays=None):
        return fedavg.validate_fit_report(self.report if report is None else report,
            self.arrays if arrays is None else arrays, self.config, [7, 9])

    def test_only_selected_hyperparameters_and_eight_neuron_architecture(self):
        self.config.validate()
        self.assertEqual(self.config.rounds, 6)
        self.assertEqual(self.config.local_config().epochs, 5)
        vector = public_initial_vector(201, self.config.local_config(), 7, -.47)
        self.assertEqual(vector.size, 1625)
        for key, value in (("local_epochs", 1), ("local_epochs", True),
                ("total_local_epochs", 300), ("learning_rate", .1), ("batch_size", 32),
                ("architecture", "deep"), ("working_variance", 2.),
                ("stopping_mode", "validation_patience"), ("momentum", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                replace(self.config, **{key: value}).validate()
        for r in range(1, 7):
            self.assertEqual(fedavg._loader_seed(self.config, 7, 1, r),
                             prior_loader_seed(PriorConfig(), 7, 1, r, "fixed_budget"))

    def test_all_six_rounds_weighted_and_previous_aggregate_rebroadcast(self):
        current = self.arrays["initial_vector"]
        weights = np.array([7., 9.]) / 16
        np.testing.assert_array_equal(self.arrays["weights"], weights)
        for r, item in enumerate(self.report["round_history"], 1):
            self.assertEqual(item["initial_sha256"], array_hash(current))
            for local in item["local"]:
                self.assertEqual(local["initial_sha256"], array_hash(current))
            expected = sum(weights[i] * self.arrays[f"round_{r}__local_{i}"] for i in range(2))
            np.testing.assert_array_equal(self.arrays[f"round_{r}__aggregate"], expected)
            current = expected
        np.testing.assert_array_equal(self.arrays["vector"], current)
        self.assertFalse(np.array_equal(current, self.arrays["initial_vector"]))
        self.assertTrue(self.validate())

    def test_fixed_budget_counts_no_selection_refit_or_convergence_claim(self):
        report = self.report
        self.assertEqual(report["training_rounds_executed"], 6)
        self.assertEqual(report["total_local_epochs_per_site"], 30)
        self.assertEqual(report["gradient_updates_per_site"], [30, 30])
        self.assertEqual(report["processed_rows_per_site"], [210, 270])
        self.assertEqual(report["stopping_reason"], "fixed_total_epochs_completed")
        self.assertIsNone(report["converged"])
        self.assertTrue(report["completion_eligible"])
        for key in ("tuning_fits", "refit_fits", "validation_fits"):
            self.assertEqual(report[key], 0)

    def test_one_site_matches_independent_repeated_existing_survival_sgd(self):
        sites = observed_sites()[:1]
        _, report, arrays = fedavg.fit_fedavg(sites, 3., 7, self.config, -.47)
        response = site_pseudo_response(sites[0].rows, 3.)
        current = public_initial_vector(5, self.config.local_config(), 7, -.47)
        for r in range(1, 7):
            _, current, _ = _train(sites[0].rows.design, response, current,
                self.config.local_config(), prior_loader_seed(PriorConfig(), 7, 0, r, "fixed_budget"))
        np.testing.assert_array_equal(arrays["vector"], current)
        self.assertTrue(fedavg.validate_fit_report(report, arrays, self.config, [7]))

    def test_pseudo_values_constructed_once_locally_and_untruncated(self):
        sites = [SurvivalSite(i, SurvivalRows(np.zeros((3, 2)), np.zeros(3), np.arange(1., 4.),
                    event, np.arange(3 * i, 3 * i + 3)))
                 for i, event in enumerate(([0, 1, 1], [1, 0, 1]))]
        with patch.object(fedavg, "site_pseudo_response", wraps=site_pseudo_response) as pseudo:
            model, report, _ = fedavg.fit_fedavg(sites, 2., 1, self.config, -.47)
        self.assertEqual(pseudo.call_count, 2)
        pv = site_pseudo_response(sites[0].rows, 2.)
        np.testing.assert_array_equal(pv, [.5, -.25, 1.25])
        self.assertEqual(report["sites"][0]["training_pseudo_sha256"], array_hash(pv))
        for item in report["round_history"]:
            self.assertEqual(item["local"][0]["training"]["training_pseudo_sha256"], array_hash(pv))
        prediction = model.predict_eta(sites[0].rows.design)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertFalse(model.vector.flags.writeable)

    def test_receiver_only_wire_counts_parameters_and_exact_payload_accounting(self):
        vectors = [np.arange(57, dtype=np.float64), np.arange(57, dtype=np.float64) + 1]
        uploads = [fedavg._serialize_upload(v, n) for v, n in zip(vectors, (7, 9))]
        vector, weights, _ = fedavg.aggregate_uploads(uploads, input_dim=5)
        np.testing.assert_array_equal(vector, weights[0] * vectors[0] + weights[1] * vectors[1])
        self.assertEqual([len(blob) for blob in uploads], [8 + 57 * 8] * 2)
        communication = self.report["communication"]
        self.assertEqual(communication["site_model_upload_count"], 12)
        self.assertEqual(communication["server_model_broadcast_count"], 12)
        self.assertEqual(communication["site_model_upload_bytes"], 12 * (8 + 57 * 8))
        self.assertEqual(communication["total_server_broadcast_bytes"], 12 * 57 * 8)
        self.assertEqual(communication["outcome_dependent_server_broadcast_phases"], 5)
        for malformed in ({"vector": vectors[0]}, uploads[0][:-1],
                struct.pack("<q", 0) + uploads[0][8:],
                struct.pack("<q", 1) + np.full(57, np.nan).astype("<f8").tobytes()):
            with self.assertRaises(ValueError):
                fedavg.aggregate_uploads([malformed], input_dim=5)
        with self.assertRaises(ValueError):
            fedavg._serialize_upload(vectors[0].astype(np.float32), 7)

    def test_saved_validation_never_fits_constructs_pseudo_values_or_infers(self):
        with patch.object(fedavg, "_train", side_effect=AssertionError), \
                patch.object(fedavg, "site_pseudo_response", side_effect=AssertionError), \
                patch.object(fedavg.MLP8Fit, "predict_eta", side_effect=AssertionError), \
                patch.object(torch.nn.Sequential, "forward", side_effect=AssertionError):
            self.assertTrue(self.validate())

    def test_rehashed_wrong_aggregate_or_broadcast_fails_semantic_validation(self):
        report, arrays = copy.deepcopy(self.report), {key: value.copy() for key, value in self.arrays.items()}
        arrays["round_2__aggregate"][0] += .01
        report["round_history"][1]["aggregate_sha256"] = array_hash(arrays["round_2__aggregate"])
        report["history_sha256"] = fedavg._history_hash(report)
        with self.assertRaises(ValueError):
            self.validate(report, arrays)
        report = copy.deepcopy(self.report)
        report["round_history"][1]["local"][0]["initial_sha256"] = array_hash(self.arrays["initial_vector"])
        report["history_sha256"] = fedavg._history_hash(report)
        with self.assertRaises(ValueError):
            self.validate(report)

    def test_budget_upload_dtype_timer_and_type_tampering_fails(self):
        changes = (
            lambda r: r.update(training_rounds_executed=True),
            lambda r: r.update(converged=True),
            lambda r: r["round_history"][0]["local"][0]["training"].update(optimizer_updates=6),
            lambda r: r["round_history"][0]["local"][0].update(upload_bytes=1),
            lambda r: r["communication"].update(training_communication_rounds=1),
            lambda r: r["timing"].update(fit_wall_seconds=0.),
            lambda r: r["round_history"][0].update(extra=0),
        )
        for change in changes:
            report = copy.deepcopy(self.report)
            change(report)
            report["history_sha256"] = fedavg._history_hash(report)
            with self.assertRaises(ValueError):
                self.validate(report)
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        arrays["round_1__local_0"] = arrays["round_1__local_0"].astype(np.float32)
        with self.assertRaises(ValueError):
            self.validate(arrays=arrays)

    def test_deterministic_numerical_history_and_observed_input_isolation(self):
        _, report, arrays = fedavg.fit_fedavg(observed_sites(), 3., 7, self.config, -.47)
        self.assertEqual(report["history_sha256"], self.report["history_sha256"])
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], self.arrays[key])
        bad_sites = observed_sites()[::-1]
        with self.assertRaises(ValueError):
            fedavg.fit_fedavg(bad_sites, 3., 7, self.config, -.47)
        with self.assertRaises(TypeError):
            fedavg.fit_fedavg(observed_sites(), 3., 7, {}, -.47)


if __name__ == "__main__":
    unittest.main()
