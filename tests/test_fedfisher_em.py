"""Tiny one-thread CPU numerical and communication-boundary fixtures."""

from dataclasses import replace
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from effect_modifier_experiment.one_shot_effects import fedfisher_em as em
from effect_modifier_experiment.one_shot_effects.fisher_models import (
    array_hash, input_coordinates, make_model, mean_categorical_fisher_diag,
    model_to_arrays, model_vector, predict_tau, restore_model, tau_and_gradient)
from effect_modifier_experiment.one_shot_effects.fisher_summary import quadratic_scores
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations
from effect_modifier_experiment.one_shot_effects.types import Rows
from effect_modifier_experiment.one_shot_effects.wire import pack_payload, unpack_payload, pad_payload


def rows(site=0, n=4):
    raw = np.linspace(-0.7, 0.8, n * 2).reshape(n, 2) + 0.1 * site
    treatment = (np.arange(n) % 2).astype(float)
    return Rows(raw, treatment, np.roll(treatment, site), np.arange(n, dtype=np.int64) + site * 100)


class FedFisherEMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **changes):
        return replace(em.FedFisherEMConfig(local_epochs=1, batch_size=4, basis_max_rows=2,
                                           rank=16, range_oversampling=2), **changes)

    def local(self, site=0, config=None, model_name="SyntheticMLP"):
        return em.fit_local(rows(site), model_name, 13, config or self.config(), site_id=site)

    def test_strict_objective_uses_site_weights_once_and_preserves_tiny_curvature(self):
        theta = np.array([[0., 1., 10.], [4., 5., 30.]])
        fisher = np.array([[0., 1e-300, 2.], [0., 3e-300, 6.]])
        actual, audit = em.strict_diagonal_solution(theta, fisher, np.array([0.25, 0.75]))
        np.testing.assert_allclose(actual, [3., 4.6, 28.], atol=1e-13)
        self.assertEqual(audit["zero_curvature_coordinates"], 1)
        self.assertEqual(audit["positive_curvature_coordinates"], 2)
        self.assertEqual(audit["aggregation_damping"], 0.)
        self.assertLess(audit["stationarity_relative_max"], 1e-12)
        with self.assertRaises(ValueError):
            em.strict_diagonal_solution(theta, -fisher, np.array([0.25, 0.75]))

    def test_fisher_is_mean_true_categorical_expectation_and_ignores_Y(self):
        train = rows(n=3)
        model = make_model("SyntheticMLP", 2, 4)
        actual = mean_categorical_fisher_diag(model, train)
        expected = np.zeros_like(actual)
        for values in input_coordinates(train.x, train.a):
            logits = model(torch.tensor(values[None], dtype=torch.float64))[0]
            log_probs = torch.log_softmax(logits, dim=0)
            probability = log_probs.exp().detach().numpy()
            for label in (0, 1):
                gradients = torch.autograd.grad(log_probs[label], tuple(model.parameters()), retain_graph=label == 0)
                vector = parameters_to_vector(gradients).detach().numpy()
                expected += probability[label] * vector * vector / len(train)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)
        changed_labels = Rows(train.x, train.a, 1 - train.y, train.row_ids)
        np.testing.assert_array_equal(actual, mean_categorical_fisher_diag(model, changed_labels))

    def test_local_training_is_reproducible_and_fedavg_can_skip_fisher(self):
        first, second = self.local(), self.local()
        np.testing.assert_array_equal(first.theta, second.theta)
        np.testing.assert_array_equal(first.fisher, second.fisher)
        other = self.local(site=1)
        self.assertEqual(first.initial_fingerprint, other.initial_fingerprint)
        with patch.object(em, "mean_categorical_fisher_diag", side_effect=AssertionError("must not compute Fisher")):
            baseline = em.fit_local(rows(), "SyntheticMLPDeep", 13, self.config(), compute_fisher=False)
        self.assertFalse(baseline.fisher_computed)
        self.assertEqual(baseline.report["fisher_status"], "not_computed")
        self.assertEqual(baseline.model.net[0].out_features, 256)
        with self.assertRaisesRegex(ValueError, "computed"):
            em.build_training_payload(baseline, 8, 0)

    def test_stage_two_strict_server_cannot_construct_low_rank_or_read_validation(self):
        config = self.config()
        local = [self.local(index, config) for index in (0, 1)]
        with patch.object(em, "build_basis", side_effect=AssertionError("stage two must not compress")):
            payloads = [em.build_training_payload(state, 8 + 2 * index, index) for index, state in enumerate(local)]
            model, report = em.aggregate_training_payloads(payloads, config)
        expected, _ = em.strict_diagonal_solution(np.stack([state.theta for state in local]),
            np.stack([state.fisher for state in local]), np.array([8 / 18, 10 / 18]))
        np.testing.assert_array_equal(model_to_arrays(model)["theta"], expected)
        self.assertEqual(report["site_weights"], [8 / 18, 10 / 18])
        self.assertEqual(report["communication"]["upload_rounds"], 1)
        self.assertEqual(report["communication"]["upload_bytes"], sum(map(len, payloads)))
        with patch.object(em, "fit_local", side_effect=AssertionError("validation cannot train")), patch.object(model, "predict_tau", side_effect=AssertionError("validation cannot predict")):
            em.validate_server_result(payloads, model, None, report, config, training_only=True)
        wrong_report = dict(report, site_weights=[0.5, 0.5])
        with self.assertRaises(ValueError):
            em.validate_server_result(payloads, model, None, wrong_report, config, training_only=True)
        self.assertEqual(set(inspect.signature(em.aggregate_training_payloads).parameters), {"payloads", "config"})
        arrays, metadata = unpack_payload(payloads[0], em.TRAINING_METHOD)
        self.assertEqual(set(arrays), {"theta", "fisher"})
        with self.assertRaises(ValueError):
            em.aggregate_training_payloads([pack_payload(em.TRAINING_METHOD, arrays, dict(metadata, validation_rows=[[1.]]))], config)
        restored = restore_model(model_to_arrays(model), report)
        np.testing.assert_array_equal(restored.predict_tau(rows().x), model.predict_tau(rows().x))

    def test_full_gradient_derivative_matches_finite_difference_and_exact_at_local_model(self):
        local = self.local()
        rng = np.random.default_rng(77)
        direction = rng.normal(size=local.theta.size)
        direction /= np.linalg.norm(direction)
        _, gradient = tau_and_gradient(local.model, rows().x[0])
        epsilon = 1e-5
        plus = restore_model({"theta": local.theta + epsilon * direction}, vars(local))
        minus = restore_model({"theta": local.theta - epsilon * direction}, vars(local))
        numerical = (plus.predict_tau(rows().x[:1])[0] - minus.predict_tau(rows().x[:1])[0]) / (2 * epsilon)
        self.assertAlmostEqual(float(gradient @ direction), numerical, places=8)
        holdout = rows(2, n=2).x
        plan = make_permutations(2, 2, 5)
        with patch.object(em, "build_basis", side_effect=AssertionError("full gradient has no basis")):
            result = em.full_gradient_diagnostics(local, local.theta, holdout, plan)
        np.testing.assert_array_equal(result["scores"]["exact"], result["scores"]["full_gradient"])
        self.assertEqual(result["tau_rmse_over_original_and_all_interventions"]["full_gradient"], 0.)

    def test_all_three_compressions_have_exact_summary_scores_for_their_approximation(self):
        holdout = rows(2, n=2).x
        plan = make_permutations(2, 2, 19)
        for compression in ("fisher_gradient", "euclidean_gradient", "random_orthogonal"):
            with self.subTest(compression=compression):
                config = self.config(compression=compression)
                local = self.local(config=config)
                payload = em.build_payload(local, holdout, plan, config, n_total=8, site_id=0)
                arrays, metadata = unpack_payload(payload, em.SCORING_METHOD)
                np.testing.assert_allclose(arrays["basis"].T @ arrays["basis"], np.eye(16), atol=1e-10)
                model, scores, report = em.aggregate([payload], config)
                diagnostic = em.simulation_diagnostics(local, payload, model_to_arrays(model)["theta"], holdout, plan)
                np.testing.assert_allclose(scores, diagnostic["scores"]["low_rank"], atol=1e-14)
                self.assertLess(diagnostic["summary_vs_direct_low_rank_max_error"], 1e-14)
                self.assertEqual(metadata["basis"]["candidate_variables"], 2)
                self.assertTrue(metadata["basis"]["matrix_free"])
                self.assertEqual(report["communication"]["upload_bytes"], len(payload))
                em.validate_server_result([payload], model, scores, report, config)
                with self.assertRaises(ValueError):
                    em.validate_server_result([payload], model, scores + 1, report, config)
                padded = pad_payload(payload, len(payload) + 100)
                padded_model, padded_scores, padded_report = em.aggregate([padded], config)
                em.validate_server_result([padded], padded_model, padded_scores, padded_report, config)
                np.testing.assert_array_equal(padded_scores, scores)
                self.assertEqual(padded_report["communication"]["upload_bytes"], len(padded))
                delta = np.linspace(-0.01, 0.01, local.theta.size)
                shifted = em.simulation_diagnostics(local, payload, local.theta + delta, holdout, plan)
                direct, _ = quadratic_scores(local.theta + delta, local.theta, local.fisher,
                    arrays["basis"], arrays["score_matrices"], config)
                np.testing.assert_allclose(direct, shifted["scores"]["low_rank"], atol=1e-12)

    def test_scoring_damping_cannot_change_strict_aggregation(self):
        first_config = self.config(score_damping=1e-6)
        second_config = replace(first_config, score_damping=10.)
        original = [self.local(index, first_config) for index in (0, 1)]
        rebound = [replace(state, config=second_config) for state in original]
        first, _ = em.aggregate_training_payloads([em.build_training_payload(state, 8, index) for index, state in enumerate(original)], first_config)
        second, _ = em.aggregate_training_payloads([em.build_training_payload(state, 8, index) for index, state in enumerate(rebound)], second_config)
        np.testing.assert_array_equal(model_to_arrays(first)["theta"], model_to_arrays(second)["theta"])

    def test_legacy_quantization_failure_and_accuracy_state_are_preserved(self):
        local = self.local()
        from utils.compress_fisher import quantize_layer
        expected_quantized = quantize_layer(torch.tensor(local.theta, dtype=torch.float64), local.model, 15).numpy()
        np.testing.assert_array_equal(em._legacy_quantize(local.theta, local.model, 15), expected_quantized)
        with self.assertRaisesRegex(FloatingPointError, "0/0"):
            em._legacy_quantize(np.zeros_like(local.theta), local.model, 15)
        payload = em.build_training_payload(local, 8, 0)
        # Avoid an architecture's genuine zero-curvature layer from hiding the
        # independent state-machine assertion: this fixture supplies positive
        # curvature and retains matching local hashes.
        arrays, metadata = unpack_payload(payload, em.TRAINING_METHOD)
        arrays = dict(arrays, fisher=np.ones_like(local.fisher))
        metadata["local_fit"]["fisher_sha256"] = array_hash(arrays["fisher"])
        payload = pack_payload(em.TRAINING_METHOD, arrays, metadata)
        values = rows().x
        logits = local.model(torch.tensor(input_coordinates(values, rows().a), dtype=torch.float64)).detach().numpy()
        validation = Rows(values, rows().a, logits.argmax(axis=1).astype(float), rows().row_ids + 1000)
        legacy_config = em.LegacyDiagnosticConfig(iterations=2, evaluate_every=1)
        selected, strict, report = em.legacy_validation_diagnostic([payload], [validation], self.config(),
            legacy_config=legacy_config)
        self.assertEqual(report["selected_zero_based_step"], 0)
        self.assertEqual(report["strict_legacy_uploaded_objective"]["method"], "strict_legacy_uploaded_objective")
        self.assertFalse(report["historical_equivalence_claim"])
        np.testing.assert_allclose(selected.predict_tau(values), strict.predict_tau(values), atol=1e-10)
        self.assertEqual(report["selected_theta_sha256"], array_hash(model_to_arrays(selected)["theta"]))
        self.assertEqual([item["zero_based_step"] for item in report["selection_history"]], [0, 1])
        self.assertFalse(report["summary_validation_limits"]["accuracy_observations_verified"])
        self.assertFalse(report["summary_validation_limits"]["iteration_trajectory_verified"])
        with patch.object(em, "legacy_validation_diagnostic", side_effect=AssertionError("must not replay optimizer")), \
             patch.object(selected.model, "forward", side_effect=AssertionError("must not infer")), \
             patch.object(strict.model, "forward", side_effect=AssertionError("must not infer")):
            em.validate_legacy_diagnostic([payload], selected, strict, report, self.config(), legacy_config)
        mutations = (
            lambda r: r["selection_history"].pop(),
            lambda r: r["selection_history"][0].__setitem__("zero_based_step", 1),
            lambda r: r.__setitem__("selected_zero_based_step", 1),  # Equal accuracies retain the first point.
            lambda r: r["selection_history"][0].__setitem__("theta_sha256", "0" * 64),
            lambda r: r.__setitem__("selected_theta_sha256", "0" * 64),
            lambda r: r.__setitem__("best_accuracy", 0.),
            lambda r: r["selection_history"][0].__setitem__("accuracy", 0.),
            lambda r: r.__setitem__("target_weights", [0.5]),
            lambda r: r.__setitem__("selection_site_weights", [0.5]),
            lambda r: r.__setitem__("selected_to_same_objective_strict_distance", 1.),
            lambda r: r.__setitem__("scope", "deployable_server"),
            lambda r: r.__setitem__("historical_equivalence_claim", True),
            lambda r: r["legacy_configuration"].__setitem__("iterations", 2000),
            lambda r: r["strict_legacy_uploaded_objective"].__setitem__("weighting", "target_pi"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = deepcopy(report)
                mutation(changed)
                with self.assertRaises(ValueError):
                    em.validate_legacy_diagnostic([payload], selected, strict, changed, self.config(), legacy_config)
        wrong_model = restore_model({"theta": model_to_arrays(strict)["theta"] + .01}, vars(local))
        with self.assertRaisesRegex(ValueError, "strict model"):
            em.validate_legacy_diagnostic([payload], selected, wrong_model, report, self.config(), legacy_config)
        zero = deepcopy(report)
        for candidate in zero["selection_history"]:
            candidate.update(accuracy=0., per_site_accuracy=[0.])
        with self.assertRaisesRegex(ValueError, "above zero"):
            em.validate_legacy_diagnostic([payload], selected, strict, zero, self.config(), legacy_config)

    def test_complete_metric_basis_recovers_full_linear_direction_without_metric_bias(self):
        fisher = np.array([0.01, 2., 100.])
        config = self.config(score_damping=0.3)
        metric = fisher + config.score_damping
        gradients = np.array([[1., 2., -1.], [-3., 1., 0.5]])
        delta_tau = np.array([0.2, -0.1])
        delta = np.array([0.1, -0.2, 0.4])
        basis = np.eye(3)
        delta_a = np.column_stack((delta_tau, gradients / np.sqrt(metric)))
        matrices = (delta_a.T @ delta_a / (2 * len(gradients)))[None]
        predicted, _ = quadratic_scores(delta, np.zeros(3), fisher, basis, matrices, config)
        expected = np.square(delta_tau + gradients @ delta).mean() / 2
        self.assertAlmostEqual(predicted[0], expected, places=14)


if __name__ == "__main__":
    unittest.main()
