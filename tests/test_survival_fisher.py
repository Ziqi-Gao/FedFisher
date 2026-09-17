"""Bounded numerical fixtures; no scientific pilot or default simulation."""

from io import BytesIO
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from effect_modifier_experiment.survival import fisher as f
from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite


class SurvivalFisherTests(unittest.TestCase):
    def test_mean_and_autograd_saturate_without_nan(self):
        eta = torch.tensor([-1000., -2., 0., 2., 1000.], dtype=torch.float64, requires_grad=True)
        mean = f._mean(eta)
        mean.sum().backward()
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.isfinite(eta.grad).all())
        self.assertEqual(float(mean[0]), 1.)
        self.assertEqual(float(mean[-1]), 0.)
        np.testing.assert_allclose(eta.grad[1:4].numpy(), -np.exp(eta.detach()[1:4].numpy()) * mean.detach()[1:4].numpy())

    def test_diagonal_matches_autograd_jacobian_not_residual_scores(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(6)
            model = nn.Sequential(nn.Linear(2, 3, dtype=torch.float64), nn.ReLU(), nn.Linear(3, 1, dtype=torch.float64))
        x = np.array([[1., -1.], [2., .2], [-.3, .4], [-2., -.1]])
        jacobians = []
        for row in x:
            output = f._mean(model(torch.tensor(row)).squeeze())
            grads = torch.autograd.grad(output, tuple(model.parameters()))
            jacobians.append(torch.cat([value.flatten() for value in grads]).numpy())
        expected = np.square(jacobians).mean(axis=0) / 2.5
        actual = f.working_fisher(model, x, method="diag", variance=2.5, batch_size=3)["diag"]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)

    def test_kfac_matches_layer_factor_definition(self):
        model = nn.Sequential(nn.Linear(2, 1, dtype=torch.float64))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[.5, -.2]], dtype=torch.float64))
            model[0].bias.fill_(.3)
        x = np.array([[1., -.4], [2., .1], [-1., 2.]])
        eta = x @ np.array([.5, -.2]) + .3
        derivative = -np.exp(eta - np.exp(eta))
        augmented = np.column_stack((x, np.ones(len(x))))
        factors = f.working_fisher(model, x, method="kfac", variance=2., batch_size=2)
        np.testing.assert_allclose(factors["a0"], augmented.T @ augmented / len(x))
        np.testing.assert_allclose(factors["g0"], [[np.square(derivative).mean() / 2.]])
        vector = np.array([.8, -.5, .2])
        dense = float(factors["g0"][0, 0]) * factors["a0"]
        np.testing.assert_allclose(f._curvature_mv(vector, factors, ((1, 2),), "kfac"), dense @ vector)
        exact = f.working_fisher(model, x, method="diag", variance=2.)["diag"]
        self.assertFalse(np.allclose(exact, np.diag(dense)))

    def test_kfac_parameter_order_includes_bias_correctly(self):
        a = np.array([[2., .3, .1], [.3, 1., .2], [.1, .2, 1.]])
        g = np.array([[1., .4], [.4, 2.]])
        vector = np.array([1., 2., 3., 4., 5., 6.])
        w = np.array([[1., 2., 5.], [3., 4., 6.]])
        expected = g @ w @ a
        actual = f._curvature_mv(vector, {"a0": a, "g0": g}, ((2, 2),), "kfac")
        np.testing.assert_allclose(actual, np.r_[expected[:, :2].ravel(), expected[:, 2]])

    def test_hidden_kfac_factors_match_autograd(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8)
            model = nn.Sequential(nn.Linear(2, 3, dtype=torch.float64), nn.ReLU(), nn.Linear(3, 1, dtype=torch.float64))
        x = np.array([[1., .2], [-.2, .3], [.4, -.5]])
        hidden_z = model[0](torch.tensor(x))
        hidden = model[1](hidden_z)
        eta = model[2](hidden)
        delta0, delta1 = torch.autograd.grad(f._mean(eta).sum(), (hidden_z, eta))
        factors = f.working_fisher(model, x, method="kfac", variance=2., batch_size=2)
        for i, (a, delta) in enumerate(((x, delta0.numpy()), (hidden.detach().numpy(), delta1.numpy()))):
            augmented = np.column_stack((a, np.ones(len(a))))
            np.testing.assert_allclose(factors[f"a{i}"], augmented.T @ augmented / len(a))
            np.testing.assert_allclose(factors[f"g{i}"], delta.T @ delta / (2. * len(a)), atol=1e-16)

    def _payload_fixture(self, method):
        config = f.FisherConfig(server_ridge=.1)
        initial = f.public_initial_vector(3, config, 0, -.5)
        vectors = [initial + .2, initial - .1]
        counts = [3, 7]
        precision = [0., 2.]
        blobs = []
        for vector, count, value in zip(vectors, counts, precision):
            if method == "diag":
                curvature = {"diag": np.full_like(initial, value)}
            else:
                curvature = {}
                for i, (o, d) in enumerate(f._shapes(3, config.architecture)):
                    curvature[f"a{i}"] = np.eye(d + 1)
                    curvature[f"g{i}"] = value * np.eye(o)
            blobs.append(f._serialize(vector, curvature, count, method, 3, config, initial))
        expected = (.3 * precision[0] * vectors[0] + .7 * precision[1] * vectors[1] + config.server_ridge * initial) / (.3 * precision[0] + .7 * precision[1] + config.server_ridge)
        return config, initial, blobs, expected

    def test_diagonal_aggregation_is_weighted_ridge_minimizer(self):
        config, initial, blobs, expected = self._payload_fixture("diag")
        actual, report = f.aggregate_fisher(blobs, initial_vector=initial, input_dim=3, config=config, method="diag")
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
        self.assertEqual(report["iterations"], 0)
        self.assertTrue(report["converged"])
        self.assertEqual(report["client_to_server_upload_phases"], 1)
        self.assertEqual(report["upload_bytes"], sum(map(len, blobs)))

    def test_kfac_cg_agrees_with_known_quadratic_solution(self):
        config, initial, blobs, expected = self._payload_fixture("kfac")
        actual, report = f.aggregate_fisher(blobs, initial_vector=initial, input_dim=3, config=config, method="kfac")
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        self.assertTrue(report["converged"])
        self.assertEqual(report["stop_reason"], "residual_tolerance")

    def test_kfac_budget_is_not_relabelled_as_convergence(self):
        config = f.FisherConfig(server_max_iterations=1, server_tolerance=1e-14)
        initial = f.public_initial_vector(3, config, 0, -.5)
        curvature = {}
        for i, (o, d) in enumerate(f._shapes(3, config.architecture)):
            curvature[f"a{i}"] = np.diag(np.arange(1., d + 2))
            curvature[f"g{i}"] = np.eye(o)
        blob = f._serialize(initial + .2, curvature, 3, "kfac", 3, config, initial)
        _, report = f.aggregate_fisher([blob], initial_vector=initial, input_dim=3, config=config, method="kfac")
        self.assertFalse(report["converged"])
        self.assertEqual(report["stop_reason"], "iteration_budget")
        self.assertEqual(report["iterations"], 1)

    def test_upload_rejects_extra_data_wrong_initialization_and_nonfinite_curvature(self):
        config, initial, blobs, _ = self._payload_fixture("diag")
        for altered in (initial + .1,):
            with self.assertRaisesRegex(ValueError, "initialization mismatch"):
                f.aggregate_fisher(blobs, initial_vector=altered, input_dim=3, config=config, method="diag")
        with np.load(BytesIO(blobs[0]), allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        values["patient_rows"] = np.ones((2, 3))
        buffer = BytesIO()
        np.savez(buffer, **values)
        with self.assertRaisesRegex(ValueError, "Unexpected payload"):
            f.aggregate_fisher([buffer.getvalue()], initial_vector=initial, input_dim=3, config=config, method="diag")
        values.pop("patient_rows")
        values["diag"][0] = np.nan
        buffer = BytesIO()
        np.savez(buffer, **values)
        with self.assertRaisesRegex(ValueError, "Invalid Fisher values"):
            f.aggregate_fisher([buffer.getvalue()], initial_vector=initial, input_dim=3, config=config, method="diag")

    def test_tiny_continuous_local_response_is_not_clipped(self):
        config = f.FisherConfig(epochs=1, batch_size=2)
        initial = f.public_initial_vector(3, config, 8, -.5)
        x = np.array([[0., .2, 0.], [1., -.3, -.3], [0., -.8, 0.], [1., .1, .1]])
        y = np.array([-.2, .7, 1.3, .4])
        blob, report = f.fit_local_fisher(x, y, initial_vector=initial, config=config, seed=3, method="diag")
        vector, count, _ = f._read_upload(blob, initial, config, 3, "diag")
        model = f._model(3, config)
        f._load_vector(model, vector)
        with torch.no_grad():
            prediction = f._mean(model(torch.tensor(x)).squeeze(-1)).numpy()
        self.assertAlmostEqual(report["final_loss"], float(np.square(prediction - y).mean() / 2), places=14)
        self.assertEqual(count, 4)
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual(report["stop_reason"], "fixed_epoch_budget")

    def test_tiny_pipeline_and_tamper_rejection(self):
        config = f.FisherConfig(epochs=1, batch_size=4)
        x = np.array([[.2, -.1], [-.3, .1], [.8, -.5], [.1, .4], [-.2, -.1], [.5, .6]])
        rows = SurvivalRows(x, np.array([0, 1, 0, 1, 0, 1]),
                            np.array([1., 2., 2.5, 4., 5., 6.]),
                            np.array([1, 0, 1, 1, 0, 1]), np.arange(6))
        site = SurvivalSite(0, rows)
        initial_eta = math.log(.12 * 3 ** 1.5)
        fitted = f.fit_fedfisher((site,), 3., config=config, seed=5, initial_eta=initial_eta)
        kwargs = dict(config=config, method="diag", seed=5, site_sizes=[6], vector=fitted.vector,
                      uploads=fitted.uploads, input_dim=5, initial_eta=initial_eta)
        self.assertTrue(f.validate_fit_report(fitted.diagnostics, **kwargs))
        with patch.object(f, "_model", wraps=f._model) as build:
            first = fitted.contrast(x)
            np.testing.assert_array_equal(first, fitted.contrast(x))
            self.assertEqual(build.call_count, 1)
        self.assertEqual(first.shape, (6,))
        with self.assertRaises(ValueError):
            fitted.vector[0] = 10.
        with self.assertRaises(ValueError):
            fitted.vector.setflags(write=True)
        with self.assertRaisesRegex(ValueError, "distinct ordered IDs"):
            f.fit_fedfisher((site, site), 3., config=config, seed=5, initial_eta=initial_eta)
        tampered = copy.deepcopy(fitted.diagnostics)
        tampered["server"]["iterations"] = 8
        with self.assertRaisesRegex(ValueError, "iteration count"):
            f.validate_fit_report(tampered, **kwargs)
        with self.assertRaisesRegex(ValueError, "stationarity"):
            f.validate_fit_report(fitted.diagnostics, **dict(kwargs, vector=fitted.vector + 1.))
        tampered = copy.deepcopy(fitted.diagnostics)
        tampered["local"][0]["site_id"] = 1
        with self.assertRaisesRegex(ValueError, "local training/upload mismatch"):
            f.validate_fit_report(tampered, **kwargs)
        tampered = copy.deepcopy(fitted.diagnostics)
        tampered["server"]["objective"] = float("nan")
        with self.assertRaisesRegex(ValueError, "aggregate diagnostics mismatch"):
            f.validate_fit_report(tampered, **kwargs)

    def test_configuration_rejects_invalid_variance_ridge_and_architecture(self):
        for config in (f.FisherConfig(working_variance=0), f.FisherConfig(server_ridge=0),
                       f.FisherConfig(architecture="binary"), f.FisherConfig(epochs=True),
                       f.FisherConfig(learning_rate=float("nan"))):
            with self.assertRaises(ValueError):
                config.validate()

    def test_fisher_rejects_nonpositive_batch_and_wrong_activation(self):
        model = nn.Sequential(nn.Linear(2, 1, dtype=torch.float64))
        for batch_size in (0, -1, True, .5):
            with self.assertRaisesRegex(ValueError, "batch_size"):
                f.working_fisher(model, np.ones((2, 2)), method="diag", batch_size=batch_size)
        model = nn.Sequential(nn.Linear(2, 2, dtype=torch.float64), nn.Sigmoid(), nn.Linear(2, 1, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "alternating Linear/ReLU"):
            f.working_fisher(model, np.ones((2, 2)), method="diag")


if __name__ == "__main__":
    unittest.main()
