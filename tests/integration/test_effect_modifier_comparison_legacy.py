"""Bounded CPU fixtures for unchanged legacy-method composition and accounting."""

from dataclasses import asdict
import copy
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.data import TensorDataset

from effect_modifier_experiment import dslassocov_benchmark as ds
from effect_modifier_experiment import one_shot_lasso_benchmark as one_shot
from effect_modifier_experiment.causal_comparison.fedavg import _make_initial_model, _model_fingerprint
from effect_modifier_experiment.causal_comparison.legacy import (
    DsCommunicationObserver, _fingerprint, _validated_ds_fit, fit_legacy, validate_fit_report,
)
from effect_modifier_experiment.causal_comparison.evaluation import model_arrays
from effect_modifier_experiment.causal_comparison.measurement import MethodTimer


def ds_config():
    return dict(family="dslassocov", optimizer=dict(
        cv_folds=2, lambda_count=2, lambda_min_ratio=0.2, cv_seed_offset=4000037,
        standardization_scale_floor=1e-12, optimizer_max_iter=2000,
        optimizer_tolerance=1e-5, optimizer_initial_step=1.0,
        optimizer_backtrack_factor=0.5, optimizer_min_step=1e-12))


def tiny_sites(rows=80, covariates=2):
    rng = np.random.default_rng(3917)
    sites = []
    for _ in range(2):
        x = rng.normal(size=(rows, covariates))
        a = rng.integers(0, 2, size=rows)
        features = np.column_stack((a, x, a[:, None] * x)).astype(np.float32)
        score = 0.25 * x[:, 0] + 0.5 * a * x[:, 0]
        y = (rng.random(rows) < 1 / (1 + np.exp(-score))).astype(np.int64)
        sites.append(TensorDataset(torch.from_numpy(features), torch.from_numpy(y)))
    return ds.tensor_datasets_to_raw_sites(sites, covariates), sites


class LegacyComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_observer_counts_actual_payloads_and_preserves_aggregates(self):
        raw = [ds.RawSite(np.array([[0., 1., 0.], [1., 2., 2.]]),
                          np.array([0., 1.]), 0),
               ds.RawSite(np.array([[1., 3., 3.], [0., -1., 0.]]),
                          np.array([0., 1.]), 1)]
        baseline_mean, baseline_scale, baseline_report = ds.aggregate_standardization(raw)
        sites = ds.standardize_sites(raw, baseline_mean, baseline_scale)
        coefficients = np.array([0.1, 0.2, -0.1, 0.3])
        plain = ds.aggregate_loss_and_gradient(sites, coefficients)
        observer = DsCommunicationObserver()
        with ds.observe_communication(observer):
            mean, scale, report = ds.aggregate_standardization(raw)
            observed = ds.aggregate_loss_and_gradient(sites, coefficients)
            loss = ds._aggregate_loss_and_gradient(sites, coefficients, False)
        np.testing.assert_array_equal(mean, baseline_mean)
        np.testing.assert_array_equal(scale, baseline_scale)
        self.assertEqual(report, baseline_report)
        self.assertEqual(observed[0], plain[0])
        self.assertEqual(loss, plain[0])
        np.testing.assert_array_equal(observed[1], plain[1])
        counts = observer.to_dict()
        self.assertEqual(counts["identified_logical_rounds"], 3)
        self.assertEqual((counts["upload_bytes"], counts["download_bytes"]), (288, 224))
        self.assertEqual(counts["messages"], 12)
        self.assertFalse(counts["actual_network_measured"])
        ds.aggregate_loss_and_gradient(sites, coefficients)
        self.assertEqual(observer.to_dict(), counts)

    def test_observer_context_is_restored_after_failure_and_nesting(self):
        raw, _ = tiny_sites(rows=4, covariates=1)
        outer, inner = DsCommunicationObserver(), DsCommunicationObserver()
        with ds.observe_communication(outer):
            ds.aggregate_standardization(raw)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with ds.observe_communication(inner):
                    ds.aggregate_standardization(raw)
                    raise RuntimeError("fixture failure")
            ds.aggregate_standardization(raw)
        self.assertEqual(outer.to_dict()["identified_logical_rounds"], 2)
        self.assertEqual(inner.to_dict()["identified_logical_rounds"], 1)
        ds.aggregate_standardization(raw)
        self.assertEqual(outer.to_dict()["identified_logical_rounds"], 2)

    def test_dslasso_wrapper_matches_unobserved_cv_and_includes_all_exchanges(self):
        raw, datasets = tiny_sites()
        config = ds_config()
        plain = ds.federated_logistic_lasso_cv(
            raw, 2, num_folds=2, lambda_count=2, lambda_min_ratio=0.2,
            cv_seed=4000037 + 9, optimizer_config=ds.OptimizerConfig(max_iter=2000, tolerance=1e-5))
        timer = MethodTimer("dslasso-fixture")
        model, audit = fit_legacy(raw, datasets, config, 9, timer)
        self.assertTrue(audit["completion_eligible"])
        self.assertEqual(audit["fit_metadata"], plain)
        np.testing.assert_array_equal(model.coefficients.detach().numpy(),
                                      np.asarray(plain["original_coefficients"], dtype=np.float32))
        kinds = audit["communication"]["event_kinds"]
        self.assertEqual(kinds["standardization"]["calls"], 3)
        self.assertGreater(kinds["loss_gradient"]["calls"],
                           plain["optimizer"]["selected_lambda_fit"]["gradient_rounds"])
        self.assertGreater(kinds["loss_only"]["calls"], 0)
        self.assertTrue(audit["timing"]["stages"]["train"]["measured"])
        self.assertFalse(audit["timing"]["stages"]["tuning"]["measured"])
        self.assertFalse(audit["timing"]["stages"]["aggregation"]["measured"])
        _, repeated = fit_legacy(raw, datasets, config, 9, MethodTimer("repeat"))
        self.assertEqual(audit["payload_fingerprint"], repeated["payload_fingerprint"])
        self.assertEqual(audit["fit_fingerprint"], repeated["fit_fingerprint"])

    def test_selected_convergence_requires_residual_evidence_not_boolean_only(self):
        report = dict(converged=True, stationarity_threshold=1e-4,
                      proximal_gradient_mapping_inf=1e-5, kkt_residual_inf=1e-5)
        fit = dict(cv=dict(selected_index=0, lambda_path=[1.0], num_folds=1,
                           selected_lambda=1.0, fold_reports=[dict(lambda_fits=[report])]),
                   optimizer=dict(selected_lambda_fit=report, final_lambda_fits=[report]))
        self.assertTrue(_validated_ds_fit(fit)["selected_final_fit_validated"])
        for invalid in (float("nan"), float("inf"), -1.0, 0.01):
            bad = copy.deepcopy(fit)
            bad["cv"]["fold_reports"][0]["lambda_fits"][0]["kkt_residual_inf"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                _validated_ds_fit(bad)

    def test_one_shot_preserves_common_initialization_and_payload_boundary(self):
        raw, datasets = tiny_sites(rows=8, covariates=1)
        config = one_shot.MlpGroupLassoConfig(covariate_dim=1, local_epochs=1, batch_size=4)
        method = dict(family="one_shot_fedavg", model="SyntheticMLP", optimizer=asdict(config))
        model, audit = fit_legacy(raw, datasets, method, 7, MethodTimer("one-shot"))
        initial = _make_initial_model("SyntheticMLP", 3, 7, None)
        vector_bytes = sum(value.numel() * value.element_size() for value in initial.state_dict().values())
        self.assertEqual(audit["initial_model_fingerprint"], _model_fingerprint(initial))
        self.assertEqual(audit["communication"]["upload_bytes"], 2 * (vector_bytes + 16))
        self.assertEqual(audit["communication"]["download_bytes"], 2 * vector_bytes)
        self.assertEqual(audit["communication"]["training_communication_rounds"], 1)
        self.assertEqual(audit["communication"]["legacy_api_report"]["post_upload_training_site_accesses"], 0)
        self.assertTrue(audit["timing"]["stages"]["aggregation"]["measured"])
        self.assertTrue(audit["completion_eligible"])
        self.assertTrue(all(np.isfinite(value.detach().numpy()).all() for value in model.parameters()))
        tampered = copy.deepcopy(audit)
        tampered["fit_metadata"]["sealed_sites"][0]["upload_count"] = 2
        # Even a self-consistent refreshed metadata hash is not scientific
        # acceptance: the semantic single-upload contract must still fail.
        tampered["fit_fingerprint"] = _fingerprint(dict(
            method_config=method, seed=7, training=tampered["training_fingerprint"],
            payload=tampered["payload_fingerprint"], model=tampered["model_fingerprint"],
            fit=tampered["fit_metadata"]))
        with self.assertRaisesRegex(ValueError, "seal or single-upload"):
            validate_fit_report(tampered, method, model_arrays(model))

    def test_local_logistic_rejects_incomplete_selected_fold_audit(self):
        raw, datasets = tiny_sites(rows=8, covariates=1)
        config = one_shot.LocalLogisticConfig(covariate_dim=1, cv_folds=2)
        method = dict(family="local_logistic_lasso", optimizer=asdict(config))

        def fake_fit(site, config, seed):
            site.seal()
            return one_shot.LinearSitePayload(site.site_idx, site.num_examples, np.zeros(4)), {
                "selected_cv_reports": [], "selected_final_report": {}, "payload_bytes": 48}

        with mock.patch.object(one_shot, "fit_local_logistic_lasso_site", side_effect=fake_fit):
            with self.assertRaisesRegex(RuntimeError, "missing local logistic CV fold"):
                fit_legacy(raw, datasets, method, 7, MethodTimer("local-logistic"))

    def test_disagreeing_data_rejected_before_method_fit(self):
        raw, datasets = tiny_sites(rows=8, covariates=1)
        raw[0].features[0, 1] += 1
        with mock.patch.object(ds, "federated_logistic_lasso_cv") as fitter:
            with self.assertRaisesRegex(ValueError, "disagree"):
                fit_legacy(raw, datasets, ds_config(), 7, MethodTimer("invalid"))
        fitter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
