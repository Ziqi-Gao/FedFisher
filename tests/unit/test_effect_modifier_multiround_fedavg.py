from copy import deepcopy
from dataclasses import replace
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from effect_modifier_experiment.causal_comparison import fedavg
from effect_modifier_experiment.causal_comparison.fedavg import FedAvgConfig, fit_multiround_fedavg
from effect_modifier_experiment.dslassocov_benchmark import RawSite


class MultiroundFedAvgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _sites(self):
        sites = []
        for site_idx, count in enumerate((10, 30)):
            raw_x = np.linspace(-1, 1, count * 2).reshape(count, 2)
            treatment = (np.arange(count) % 2).astype(float)
            features = np.column_stack((treatment, raw_x, treatment[:, None] * raw_x))
            sites.append(RawSite(features, treatment, site_idx))
        return sites

    def _config(self, **kwargs):
        return replace(FedAvgConfig(covariate_dim=2, batch_size=8, min_rounds=1, max_rounds=5, patience=2), **kwargs)

    @staticmethod
    def _constant_validation(model, sites):
        return 1.0, [{"site_idx": site.site_idx, "loss_sum": float(site.labels.numel()), "num_examples": int(site.labels.numel())} for site in sites]

    def test_second_round_broadcasts_weighted_aggregate_and_budget_uses_all_rows(self):
        observed = []

        def known_updates(global_model, site, config, loader_seed):
            observed.append(parameters_to_vector(global_model.parameters()).detach().clone())
            model = deepcopy(global_model)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(float(site.site_idx + 1))
            return model, config.local_epochs * math.ceil(site.labels.numel() / config.batch_size)

        with patch.object(fedavg, "_train_local_model", side_effect=known_updates):
            model, audit = fit_multiround_fedavg(self._sites(), "SyntheticMLP", self._config(stopping_mode="fixed_total_epochs", total_local_epochs=2), 17)
        self.assertTrue(torch.equal(observed[0], observed[1]))
        torch.testing.assert_close(observed[2], observed[0] + 1.75)
        self.assertTrue(torch.equal(observed[2], observed[3]))
        torch.testing.assert_close(parameters_to_vector(model.parameters()), observed[0] + 3.5)
        self.assertEqual(audit["round_history"][0]["aggregation_weights"], [0.25, 0.75])
        self.assertEqual(audit["round_history"][1]["initial_model_fingerprint"], audit["round_history"][0]["aggregate_model_fingerprint"])
        self.assertEqual(audit["training_examples_seen_per_site"], [20, 60])
        self.assertEqual(audit["gradient_updates_per_site"], [4, 8])
        self.assertEqual(audit["termination_reason"], "fixed_total_epochs_completed")
        self.assertTrue(audit["completion_eligible"])
        self.assertIsNone(audit["converged"])
        comm = audit["communication"]
        vector_bytes = parameters_to_vector(model.parameters()).numel() * 4
        self.assertEqual(comm["optimization_rounds"], 2)
        self.assertEqual(comm["validation_communication_rounds"], 0)
        self.assertEqual(comm["site_model_upload_count"], 4)
        self.assertEqual(comm["total_server_broadcast_bytes"], 4 * vector_bytes)
        self.assertEqual(comm["total_site_upload_bytes"], 4 * (vector_bytes + 16))

    def test_patience_selects_then_refits_full_rows_from_initialization(self):
        with patch.object(fedavg, "_validation_loss", side_effect=self._constant_validation):
            _, audit = fit_multiround_fedavg(self._sites(), "SyntheticMLP", self._config(), 23)
        self.assertEqual(audit["termination_reason"], "validation_patience_satisfied")
        self.assertTrue(audit["converged"])
        self.assertEqual(audit["best_selection_round"], 1)
        self.assertEqual(audit["refit_rounds"], 1)
        self.assertEqual(audit["training_rounds_executed"], 4)
        history = audit["round_history"]
        self.assertEqual([row["phase"] for row in history], ["selection"] * 3 + ["refit"])
        self.assertEqual(history[-1]["initial_model_fingerprint"], audit["initial_model_fingerprint"])
        self.assertEqual([row["num_examples"] for row in history[-1]["sites"]], [10, 30])
        self.assertEqual([row["num_examples"] for row in history[0]["sites"]], [8, 24])
        self.assertEqual(audit["communication"]["total_synchronization_phases"], 7)
        self.assertEqual(audit["communication"]["site_validation_upload_count"], 6)
        self.assertEqual(audit["communication"]["site_validation_upload_bytes"], 6 * 24)

    def test_all_communication_frequencies_preserve_the_thirty_epoch_budget(self):
        def counted_updates(global_model, site, config, loader_seed):
            return deepcopy(global_model), config.local_epochs * math.ceil(site.labels.numel() / config.batch_size)

        for epochs in (1, 5, 10, 30):
            with self.subTest(local_epochs=epochs), patch.object(fedavg, "_train_local_model", side_effect=counted_updates):
                _, audit = fit_multiround_fedavg(
                    self._sites()[:1], "SyntheticMLP",
                    self._config(stopping_mode="fixed_total_epochs", total_local_epochs=30, local_epochs=epochs, max_rounds=30),
                    12,
                )
            self.assertEqual(audit["training_rounds_executed"], 30 // epochs)
            self.assertEqual(audit["total_local_epochs_per_site"], 30)
            self.assertEqual(audit["gradient_updates_per_site"], [60])
            self.assertEqual(audit["training_examples_seen_per_site"], [300])
            self.assertTrue(audit["completion_eligible"])

    def test_safety_limit_cannot_report_convergence_or_trigger_refit(self):
        with patch.object(fedavg, "_validation_loss", side_effect=self._constant_validation):
            _, audit = fit_multiround_fedavg(self._sites(), "SyntheticMLP", self._config(max_rounds=2), 5)
        self.assertFalse(audit["completion_eligible"])
        self.assertFalse(audit["converged"])
        self.assertTrue(audit["safety_limit_reached"])
        self.assertEqual(audit["refit_rounds"], 0)
        self.assertEqual(audit["training_rounds_executed"], 2)
        self.assertEqual(audit["returned_model_phase"], "diagnostic_nonconverged_checkpoint")

    def test_fixed_epoch_safety_exhaustion_is_not_completion(self):
        _, audit = fit_multiround_fedavg(self._sites(), "SyntheticMLP", self._config(stopping_mode="fixed_total_epochs", total_local_epochs=30, max_rounds=1), 1)
        self.assertFalse(audit["completion_eligible"])
        self.assertEqual(audit["termination_reason"], "maximum_rounds_reached")
        self.assertEqual(audit["total_local_epochs_per_site"], 1)

    def test_identical_seed_is_deterministic_without_changing_callers_rng(self):
        config = self._config(stopping_mode="fixed_total_epochs", total_local_epochs=2)
        torch.random.default_generator.manual_seed(818)
        state = torch.random.get_rng_state().clone()
        first, first_audit = fit_multiround_fedavg(self._sites(), "SyntheticMLP", config, 91)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        second, second_audit = fit_multiround_fedavg(self._sites(), "SyntheticMLP", config, 91)
        self.assertTrue(torch.equal(parameters_to_vector(first.parameters()), parameters_to_vector(second.parameters())))
        self.assertEqual(first_audit["history_fingerprint"], second_audit["history_fingerprint"])
        self.assertEqual(first_audit["communication"], second_audit["communication"])
        self.assertGreater(first_audit["timing"]["fit_wall_seconds"], 0)

    def test_validation_is_weighted_by_heldout_rows_and_uses_no_test_input(self):
        config = self._config()
        _, _, validation_sites, _ = fedavg._prepare_sites(self._sites(), config, 8)
        model = fedavg._make_initial_model("SyntheticMLP", 5, 8, None)
        loss, _ = fedavg._validation_loss(model, validation_sites)
        with torch.no_grad():
            all_features = torch.cat([site.features for site in validation_sites])
            all_labels = torch.cat([site.labels for site in validation_sites])
            expected = float(torch.nn.functional.cross_entropy(model(all_features), all_labels))
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertEqual(set(inspect.signature(fit_multiround_fedavg).parameters), {"raw_sites", "model_name", "config", "seed", "initial_state"})
        with self.assertRaises(TypeError):
            fit_multiround_fedavg(self._sites(), "SyntheticMLP", config, 8, test_dataset=object())

    def test_deep_architecture_and_observed_single_class_sites_are_supported(self):
        sites = [RawSite(site.features, np.ones(site.labels.size), site.site_idx) for site in self._sites()]
        with patch.object(fedavg, "_validation_loss", side_effect=self._constant_validation):
            model, audit = fit_multiround_fedavg(sites, "SyntheticMLPDeep", self._config(max_rounds=2, patience=1, refit_full_data=False), 3)
        self.assertTrue(audit["completion_eligible"])
        self.assertEqual(audit["refit_rounds"], 0)
        self.assertEqual(model.net[0].out_features, 256)
        self.assertEqual(audit["site_splits"][0]["label_counts"], {"0": 0, "1": 10})

    def test_config_and_input_validation_are_strict(self):
        for changes in (
            {"local_epochs": True}, {"local_epochs": 2}, {"learning_rate": float("nan")},
            {"momentum": True}, {"batch_size": 0}, {"patience": False},
            {"refit_full_data": 1}, {"stopping_mode": "forever"},
            {"min_rounds": 3, "max_rounds": 2}, {"min_delta": -1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._config(**changes)
        sites = self._sites()
        with self.assertRaises(ValueError):
            fit_multiround_fedavg(sites, "SyntheticMLP", self._config(), True)
        with self.assertRaises(ValueError):
            fit_multiround_fedavg(sites[::-1], "SyntheticMLP", self._config(), 1)
        bad = [RawSite(np.full((10, 5), float("nan")), sites[0].labels, 0)]
        with self.assertRaises(ValueError):
            fit_multiround_fedavg(bad, "SyntheticMLP", self._config(), 1)


if __name__ == "__main__":
    unittest.main()
