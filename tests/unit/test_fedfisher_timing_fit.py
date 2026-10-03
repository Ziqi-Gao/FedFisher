"""Tiny observed-data fixtures for independent, measured CPU Fisher methods."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import random
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import TensorDataset

from models import SyntheticMLP
from nngeometry.object import PMatDiag, PMatKFAC
from train_model import LocalUpdate
from effect_modifier_experiment.causal_comparison.evaluation import model_arrays
from effect_modifier_experiment.causal_comparison.measurement import MethodTimer
from effect_modifier_experiment.dslassocov_benchmark import dataset_fingerprint
from effect_modifier_experiment.fedfisher_timing import fit


def recipe(algorithm="fedfisher_diag"):
    return dict(family="legacy_fedfisher", algorithm=algorithm, model="SyntheticMLP",
                optimizer=dict(local_epochs=1, batch_size=4, learning_rate=0.01,
                    momentum=0.9, validation_size=7, server_iterations=2000,
                    server_learning_rate=0.01, validation_interval=100,
                    quantization_bits=15, rng_policy=fit.RNG_POLICY))


def sites_fixture():
    rng = np.random.default_rng(443)
    sites = []
    for n in (12, 16):
        x = rng.normal(size=(n, 1)).astype(np.float32)
        a = (np.arange(n) % 2).astype(np.float32)[:, None]
        values = np.column_stack((a, x, a*x)).astype(np.float32)
        labels = (np.arange(n) % 2).astype(np.int64)
        sites.append(TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)))
    # Preserve a non-site row ordering to test the old global validation policy.
    pooled = [torch.cat([site.tensors[i] for site in sites], 0) for i in (0, 1)]
    order = torch.arange(len(pooled[0]) - 1, -1, -1)
    return sites, TensorDataset(*(tensor[order] for tensor in pooled))


def tiny_model(model, n_c, bias=False, synthetic_dim=3):
    return SyntheticMLP(synthetic_dim, hidden_dims=(4,), n_out=n_c)


def rng_snapshot():
    numpy = np.random.get_state()
    return torch.random.get_rng_state().clone(), (numpy[0], numpy[1].copy(), *numpy[2:]), random.getstate()


class FedFisherTimingFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_rng_equal(self, before):
        after = rng_snapshot()
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertEqual(before[1][0], after[1][0])
        np.testing.assert_array_equal(before[1][1], after[1][1])
        self.assertEqual(before[1][2:], after[1][2:])
        self.assertEqual(before[2], after[2])

    def run_fit(self, method, timer=None, **kwargs):
        sites, global_train = sites_fixture()
        with mock.patch.object(fit, "get_model", side_effect=tiny_model), redirect_stdout(io.StringIO()):
            return fit.fit_fedfisher(sites, method, 9, timer,
                global_training=kwargs.pop("global_training", global_train), **kwargs)

    def test_real_cpu_diag_timing_is_observational_and_restores_rng(self):
        before = rng_snapshot()
        plain, audit = self.run_fit(recipe())
        self.assert_rng_equal(before)
        timer = MethodTimer("fedfisher_diag_shallow")
        observed, observed_audit = self.run_fit(recipe(), timer)
        self.assert_rng_equal(before)
        self.assertEqual(audit, observed_audit)
        for name, values in model_arrays(plain).items():
            np.testing.assert_array_equal(values, model_arrays(observed)[name])
        stages = timer.to_dict()["stages"]
        self.assertEqual(stages["train"]["calls"], 1)
        self.assertEqual(stages["aggregation"]["calls"], 1)
        self.assertEqual(set(stages["train"]["components"]), {"sgd", "fisher_diag"})
        self.assertTrue(all(item["calls"] == 2 for item in stages["train"]["components"].values()))
        self.assertFalse(stages["tuning"]["measured"])
        self.assertFalse(stages["evaluation"]["measured"])
        self.assertTrue(audit["completion_eligible"])
        self.assertFalse(audit["convergence_claim"])
        self.assertEqual(audit["total_local_updates"], 7)
        self.assertEqual(audit["total_sample_visits"], 28)

    def test_real_compressed_kfac_cpu_and_same_sgd_as_diag(self):
        before = rng_snapshot()
        _, diag = self.run_fit(recipe())
        timer = MethodTimer("fedfisher_kfac_shallow")
        model, kfac = self.run_fit(recipe("fedfisher_kfac"), timer)
        self.assert_rng_equal(before)
        self.assertEqual(diag["local_model_fingerprints"], kfac["local_model_fingerprints"])
        self.assertEqual(diag["initial_model_fingerprint"], kfac["initial_model_fingerprint"])
        self.assertEqual(diag["validation"], kfac["validation"])
        self.assertTrue(all(np.isfinite(value).all() for value in model_arrays(model).values()))
        self.assertEqual(set(timer.to_dict()["stages"]["train"]["components"]), {"sgd", "fisher_kfac"})
        self.assertEqual(kfac["server_iterations"], 2000)
        self.assertEqual(kfac["validation_checks"], 20)

    def test_selected_fisher_only_and_default_both_order(self):
        site = sites_fixture()[0][0]
        args = dict(bs=4, local_epochs=0, eta=0.01, device="cpu")
        calls = []

        def fim(**kwargs):
            calls.append((kwargs["representation"], kwargs["device"]))
            # Deliberately consume differing RNG, which selected mode isolates.
            torch.rand(3 if kwargs["representation"] is PMatDiag else 7)
            return SimpleNamespace(get_diag=lambda: torch.ones(26))

        with mock.patch("train_model.FIM", side_effect=fim):
            local = LocalUpdate(args, site)
            model = tiny_model("SyntheticMLP", 2)
            before = rng_snapshot()
            _, _, kfac, diag = local.train_and_compute_fisher(
                model, 2, fisher_kind="diag", isolate_fisher_rng=True)
            self.assert_rng_equal(before)
            self.assertIsNone(kfac)
            self.assertIsNotNone(diag)
            self.assertEqual(calls, [(PMatDiag, "cpu")])
            calls.clear()
            local.train_and_compute_fisher(model, 2)
            self.assertEqual(calls, [(PMatKFAC, "cpu"), (PMatDiag, "cpu")])

    def test_original_global_validation_rows_are_preserved(self):
        sites, global_train = sites_fixture()
        _, audit = self.run_fit(recipe())
        selected = np.random.RandomState(3).choice(len(global_train), 7, replace=False)
        validation = TensorDataset(*(tensor[selected] for tensor in global_train.tensors))
        self.assertEqual(audit["validation"]["dataset_sha256"], dataset_fingerprint(validation))
        self.assertEqual(audit["validation"]["pooled_order"], "original_generator_global_row_order")
        self.assertEqual(audit["validation"]["checked_iteration_indices"], list(range(0, 2000, 100)))
        self.assertEqual(audit["training_communication_rounds"], 1)
        self.assertFalse(audit["communication"]["actual_network_measured"])

    def test_foreign_global_rows_are_rejected_before_training(self):
        sites, global_train = sites_fixture()
        values, labels = (tensor.clone() for tensor in global_train.tensors)
        labels[0] = 1 - labels[0]
        before = rng_snapshot()
        with mock.patch.object(fit, "get_model") as factory:
            with self.assertRaisesRegex(ValueError, "exactly the local training union"):
                fit.fit_fedfisher(sites, recipe(), 9,
                    global_training=TensorDataset(values, labels))
            factory.assert_not_called()
        self.assert_rng_equal(before)

    def test_failure_restores_rng_and_is_recorded_as_failure(self):
        timer = MethodTimer("failure")
        before = rng_snapshot()
        with mock.patch("train_model.FIM", side_effect=RuntimeError("fixture Fisher failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture Fisher failure"):
                self.run_fit(recipe(), timer)
        self.assert_rng_equal(before)
        self.assertEqual(timer.to_dict()["stages"]["train"]["failed_calls"], 1)
        self.assertFalse(timer.to_dict()["stages"]["aggregation"]["measured"])

    def test_invalid_recipes_and_data_fail_before_fitting(self):
        sites, _ = sites_fixture()
        for field, value in (("momentum", 0.8), ("server_iterations", 100),
                             ("validation_interval", 10), ("quantization_bits", 8),
                             ("rng_policy", "legacy"), ("local_epochs", True),
                             ("server_iterations", 2000.0)):
            invalid = recipe()
            invalid["optimizer"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                fit.fit_fedfisher(sites, invalid, 9)
        invalid_sites = deepcopy(sites)
        invalid_sites[0].tensors[0][0, 2] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            fit.fit_fedfisher(invalid_sites, recipe(), 9)

    def test_nonfinite_fisher_or_model_never_yields_a_success_audit(self):
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            fit._fisher_hash(torch.tensor([float("nan")]), "diag")
        with self.assertRaisesRegex(FloatingPointError, "nonnegative"):
            fit._fisher_hash(torch.tensor([-1.0]), "diag")
        with mock.patch.object(fit, "_fisher_hash", side_effect=FloatingPointError("bad Fisher")):
            with self.assertRaisesRegex(FloatingPointError, "bad Fisher"):
                self.run_fit(recipe())
        model = tiny_model("SyntheticMLP", 2)
        with torch.no_grad():
            next(model.parameters())[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "nonfinite model"):
            fit._finite_model_hash(model)


if __name__ == "__main__":
    unittest.main()
