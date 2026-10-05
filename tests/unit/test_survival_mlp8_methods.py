"""Tiny observed fixtures; no default-dimension scientific experiment runs."""

import copy
from dataclasses import replace
from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite
from effect_modifier_experiment.survival.fisher import ARCHITECTURES, _curvature_mv, _mean, working_fisher
from effect_modifier_experiment.survival_mlp8.methods import (
    MLP8Config, _model, _read_upload, _responses, _serialize, _train,
    aggregate_fisher, fit_federated, fit_pooled, public_initial_vector,
    validate_federated_fit, validate_pooled_fit,
)


def sites():
    rng = np.random.default_rng(381)
    return [SurvivalSite(index, SurvivalRows(rng.normal(size=(n, 2)), np.arange(n) % 2,
            np.resize([1., 2., 4., 5.], n), np.resize([1, 0, 1, 1], n),
            np.arange(index * 100, index * 100 + n))) for index, n in enumerate((7, 9))]


class MLP8MethodsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = MLP8Config(epochs=2, batch_size=4)
        self.eta = -.47

    def test_exact_fixed_architecture_and_public_init_preserve_rng(self):
        before = torch.random.get_rng_state().clone()
        model = _model(201, self.config, 41)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1625)
        self.assertEqual([layer.out_features for layer in model if isinstance(layer, torch.nn.Linear)], [8, 1])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(ARCHITECTURES, {"shallow": (64, 32), "deep": (256, 128, 64, 32)})
        a = public_initial_vector(5, self.config, 3, self.eta)
        np.testing.assert_array_equal(a, public_initial_vector(5, self.config, 3, self.eta))
        self.assertEqual(a[-1], self.eta)
        with self.assertRaises(ValueError):
            replace(self.config, architecture="deep").validate()

    def test_working_fisher_matches_individual_mean_gradients(self):
        model = _model(5, self.config, 4)
        x = sites()[0].rows.design
        grads = []
        for row in x:
            model.zero_grad()
            _mean(model(torch.from_numpy(row))).squeeze().backward()
            grads.append(parameters_to_vector([p.grad for p in model.parameters()]).numpy().copy())
        expected = np.square(grads).mean(axis=0)
        np.testing.assert_allclose(working_fisher(model, x, method="diag")["diag"], expected,
                                   rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(working_fisher(model, x, method="diag", variance=3.)["diag"],
                                   expected / 3., rtol=1e-12, atol=1e-15)

    def test_weighted_server_matches_independent_dense_objective(self):
        config = replace(self.config, server_tolerance=1e-12)
        initial = public_initial_vector(5, config, 3, self.eta)
        rng = np.random.default_rng(987)
        for kind in ("diag", "kfac"):
            local = [initial + rng.normal(scale=.01, size=len(initial)) for _ in range(2)]
            if kind == "diag":
                curvature = [{"diag": rng.uniform(.1, 1., size=len(initial))} for _ in range(2)]
            else:
                curvature = []
                for _ in range(2):
                    factors = {}
                    for index, (out, inp) in enumerate(((8, 5), (1, 8))):
                        for name, size in ((f"a{index}", inp + 1), (f"g{index}", out)):
                            raw = rng.normal(size=(size, size))
                            factors[name] = raw @ raw.T + np.eye(size)
                    curvature.append(factors)
            blobs = [_serialize(v, f, n, kind, 5, config, initial)
                     for v, f, n in zip(local, curvature, (7, 9))]
            vector, report, _, _ = aggregate_fisher(blobs, initial_vector=initial,
                input_dim=5, config=config, kind=kind)
            matrices = [np.column_stack([_curvature_mv(basis, f, ((8, 5), (1, 8)), kind)
                         for basis in np.eye(len(initial))]) for f in curvature]
            precision = config.server_ridge * np.eye(len(initial)) + (7 * matrices[0] + 9 * matrices[1]) / 16
            rhs = config.server_ridge * initial + (7 * matrices[0] @ local[0] + 9 * matrices[1] @ local[1]) / 16
            np.testing.assert_allclose(vector, np.linalg.solve(precision, rhs), rtol=1e-8, atol=1e-10)
            self.assertTrue(report["converged"])
            self.assertLessEqual(np.linalg.norm(precision @ vector - rhs), report["residual_threshold"] * 1.01)

    def test_pseudo_values_untruncated_and_scope_differences(self):
        observed = [SurvivalSite(i, SurvivalRows(np.zeros((3, 2)), np.zeros(3), np.arange(1., 4.),
                    event, np.arange(3 * i, 3 * i + 3)))
                    for i, event in enumerate(([0, 1, 1], [1, 0, 1]))]
        local, pooled = _responses(observed, 2., "site_local_pv"), _responses(observed, 2., "pooled_km_pv")
        np.testing.assert_array_equal(local[0], [.5, -.25, 1.25])
        self.assertFalse(np.array_equal(np.concatenate(local), np.concatenate(pooled)))

    def test_one_site_pooled_matches_local_sgd_and_computes_no_fisher(self):
        observed = sites()[:1]
        with patch("effect_modifier_experiment.survival_mlp8.methods.working_fisher", side_effect=AssertionError):
            fitted, report, arrays = fit_pooled(observed, 3., 7, "site_local_pv", self.config, self.eta)
        _, local, local_report = _train(observed[0].rows.design, _responses(observed, 3., "site_local_pv")[0],
            public_initial_vector(5, self.config, 7, self.eta), self.config, 7 + 1009)
        np.testing.assert_array_equal(arrays["vector"], local)
        self.assertEqual(report["final_loss"], local_report["final_loss"])
        self.assertEqual(report["optimizer_updates"], 4)
        self.assertIs(report["fisher_computed"], False)
        self.assertTrue(validate_pooled_fit(report, arrays, "site_local_pv", self.config, [7]))
        self.assertEqual(fitted.predict_eta(observed[0].rows.design).shape, (7,))
        self.assertFalse(fitted.vector.flags.writeable)

    def test_diag_kfac_local_vectors_match_and_saved_validators_do_not_infer(self):
        fits = {}
        for kind in ("diag", "kfac"):
            model, report, arrays = fit_federated(sites(), 3., 2, kind, self.config, self.eta)
            with patch("effect_modifier_experiment.survival_mlp8.methods.MLP8Fit.predict_eta", side_effect=AssertionError):
                self.assertTrue(validate_federated_fit(report, arrays, kind, self.config, [7, 9]))
            self.assertEqual(report["server"]["client_to_server_upload_phases"], 1)
            self.assertEqual(report["server"]["uploads"], 2)
            fits[kind] = arrays
        for key in ("initial_vector", "local_0__vector", "local_1__vector"):
            np.testing.assert_array_equal(fits["diag"][key], fits["kfac"][key])
        np.testing.assert_array_equal(fits["diag"]["weights"], np.array([7., 9.]) / 16)

    def test_serialized_server_rejects_nonwire_and_bad_curvature(self):
        initial = public_initial_vector(5, self.config, 0, self.eta)
        curvature = {"diag": np.ones_like(initial)}
        blob = _serialize(initial, curvature, 7, "diag", 5, self.config, initial)
        self.assertEqual(blob, _serialize(initial, curvature, 7, "diag", 5, self.config, initial))
        _read_upload(blob, initial, self.config, 5, "diag")
        with self.assertRaises(ValueError):
            aggregate_fisher([{"vector": initial}], initial_vector=initial, input_dim=5, config=self.config, kind="diag")
        with self.assertRaises(ValueError):
            _read_upload(blob, initial + 1, self.config, 5, "diag")
        with self.assertRaises(ValueError):
            _read_upload(_serialize(initial, {"diag": -curvature["diag"]}, 7, "diag", 5,
                                   self.config, initial), initial, self.config, 5, "diag")
        with np.load(BytesIO(blob), allow_pickle=False) as saved:
            fields = {key: saved[key] for key in saved.files}
        fields["vector"] = initial.astype(np.float32)
        stream = BytesIO()
        np.savez(stream, **fields)
        with self.assertRaises(ValueError):
            _read_upload(stream.getvalue(), initial, self.config, 5, "diag")
        fields["vector"] = initial
        raw = fields["metadata"].tobytes().decode()
        fields["metadata"] = np.frombuffer((raw[:-1] + ',"kind":"diag"}').encode(), dtype=np.uint8)
        stream = BytesIO()
        np.savez(stream, **fields)
        with self.assertRaises(ValueError):
            _read_upload(stream.getvalue(), initial, self.config, 5, "diag")

    def test_rehashed_saved_endpoint_and_budget_tampering_rejected(self):
        for kind in ("diag", "kfac"):
            _, report, arrays = fit_federated(sites(), 3., 1, kind, self.config, self.eta)
            for key in ("weights", "vector", "initial_vector"):
                altered = {k: v.copy() for k, v in arrays.items()}
                altered[key][0] += .1
                with self.assertRaises(ValueError):
                    validate_federated_fit(report, altered, kind, self.config, [7, 9])
            altered_report = copy.deepcopy(report)
            altered_report["local"][0]["optimizer_updates"] += 1
            with self.assertRaises(ValueError):
                validate_federated_fit(altered_report, arrays, kind, self.config, [7, 9])
            altered_report = copy.deepcopy(report)
            altered_report["server"]["residual_norm"] = 10.
            with self.assertRaises(ValueError):
                validate_federated_fit(altered_report, arrays, kind, self.config, [7, 9])
            for location, key, value in ((None, "extra", 1), ("local", "extra", 1),
                                         ("server", "extra", 1), ("server", "client_to_server_upload_phases", True)):
                altered_report = copy.deepcopy(report)
                target = altered_report if location is None else (
                    altered_report["local"][0] if location == "local" else altered_report["server"])
                target[key] = value
                with self.assertRaises(ValueError):
                    validate_federated_fit(altered_report, arrays, kind, self.config, [7, 9])
        for scope in ("site_local_pv", "pooled_km_pv"):
            _, report, arrays = fit_pooled(sites(), 3., 1, scope, self.config, self.eta)
            self.assertTrue(validate_pooled_fit(report, arrays, scope, self.config, [7, 9]))
            altered_report = copy.deepcopy(report)
            altered_report["processed_rows"] += 1
            with self.assertRaises(ValueError):
                validate_pooled_fit(altered_report, arrays, scope, self.config, [7, 9])
            altered_report = copy.deepcopy(report)
            altered_report["communication_uploads"] = False
            with self.assertRaises(ValueError):
                validate_pooled_fit(altered_report, arrays, scope, self.config, [7, 9])


if __name__ == "__main__":
    unittest.main()
