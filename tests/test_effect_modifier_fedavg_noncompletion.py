"""Bounded, one-thread fixtures for preserved 200-round stopping semantics."""

from copy import deepcopy
from functools import lru_cache
import resource
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from effect_modifier_experiment.causal_comparison import fedavg, runner
from effect_modifier_experiment.causal_comparison.evaluation import model_arrays
from effect_modifier_experiment.causal_comparison.measurement import MethodTimer
from effect_modifier_experiment.causal_comparison.noncompletion import (
    MEMORY_SCOPE, OUTCOME, ScientificNoncompletion, validate_capped_fit, validate_noncompletion_result)
from effect_modifier_experiment.dslassocov_benchmark import tensor_datasets_to_raw_sites
from effect_modifier_experiment.infrastructure.artifacts import digest
from effect_modifier_experiment.model_signal_ablation import protocol


def tiny_datasets():
    sites = [TensorDataset(torch.zeros((6, 5)), torch.ones(6, dtype=torch.long)) for _ in range(2)]
    test = TensorDataset(torch.zeros((4, 5)), torch.ones(4, dtype=torch.long))
    return sites, test, {"0": {"1": 6}, "1": {"1": 6}}


@lru_cache(maxsize=2)
def _cached_capped_fixture(seed=25):
    torch.set_num_threads(1)
    config = protocol.run_config(protocol.FORMAL_TASK, dict(setting="small_mlp",
        cell="fedavg_small_e1_patience", variant="control", shard_index=0, shard_count=16))
    config["dgp"].update(num_sites=2, covariate_dim=2, site_min_samples=6, site_max_samples=6, num_test=4)
    config["method"]["optimizer"]["covariate_dim"] = 2
    # Only the input fixture is replaced. This runs the unchanged real SGD,
    # aggregation, holdout, patience and diagnostic checkpoint code 200 times.
    with patch.object(runner, "generate_datasets", return_value=tiny_datasets()):
        try:
            runner.compute_cell(config, seed, "noniid")
        except ScientificNoncompletion as error:
            result, arrays = error.result, error.arrays
        else:
            raise AssertionError("tiny fixture unexpectedly satisfied validation patience")
    result["identity"] = dict(run_id=config["run_id"], config_sha256=digest(config),
        provenance_sha256="a" * 64, seed=seed, split="noniid", cell=config["cell"], variant=config["variant"])
    result["memory"] = dict(process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            scope=MEMORY_SCOPE)
    validate_noncompletion_result(result, arrays, result["identity"], config)
    return config, result, arrays


def capped_fixture(seed=25):
    """Return detached config/result/arrays for recovery-store contract tests."""
    return deepcopy(_cached_capped_fixture(seed))


def rehash_fit(audit):
    """Make tampered evidence self-consistent at hash level for semantic tests."""
    material = dict(schema_version=1, model_name=audit["model_name"], seed=audit["seed"],
        config=audit["config"], site_splits=audit["site_splits"],
        initial_model_fingerprint=audit["initial_model_fingerprint"],
        history=fedavg._without_timings(audit["round_history"]),
        returned_model_fingerprint=audit["model_fingerprint"], termination_reason=audit["termination_reason"])
    audit["history_fingerprint"] = audit["fit_fingerprint"] = digest(material)


class FedAvgNoncompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.result, cls.arrays = capped_fixture()

    def validate(self, result=None, arrays=None, config=None, identity=None):
        result = self.result if result is None else result
        config = self.config if config is None else config
        arrays = self.arrays if arrays is None else arrays
        identity = self.result["identity"] if identity is None else identity
        validate_noncompletion_result(result, arrays, identity, config)

    def test_real_cap_is_diagnostic_with_full_cost_and_no_evaluation(self):
        self.validate()
        result, fit = self.result, self.result["fit"]
        self.assertEqual(result["status"], "scientific_noncompletion")
        self.assertEqual(result["outcome"], OUTCOME)
        self.assertNotIn("evaluation", result)
        self.assertFalse(fit["completion_eligible"])
        self.assertFalse(fit["converged"])
        self.assertEqual(fit["training_rounds_executed"], 200)
        self.assertEqual(fit["refit_rounds"], 0)
        self.assertEqual(fit["communication"]["total_synchronization_phases"], 400)
        self.assertEqual(fit["gradient_updates_per_site"], [200, 200])
        self.assertEqual(result["timing"]["stages"]["train"]["failed_calls"], 0)
        self.assertFalse(result["timing"]["stages"]["evaluation"]["measured"])
        with self.assertRaises(ValueError):
            fedavg.validate_fit_report(fit, self.config["method"]["optimizer"],
                                      "SyntheticMLPSmall", 25, self.arrays)

    def test_seed_zero_is_also_a_real_bounded_cap_fixture(self):
        config, result, arrays = capped_fixture(seed=0)
        self.assertEqual(result["identity"]["seed"], 0)
        self.assertEqual(result["fit"]["seed"], 0)
        self.assertEqual(result["fit"]["training_rounds_executed"], 200)
        validate_noncompletion_result(result, arrays, result["identity"], config)

    def test_semantic_tampering_fails_after_history_is_rehashed(self):
        mutations = {
            "stale": lambda a: a["round_history"][40].update(stale_rounds=True),
            "refit": lambda a: a.update(refit_rounds=1),
            "returned phase": lambda a: a.update(returned_model_phase="full_data_refit"),
            "best round": lambda a: a.update(best_selection_round=1, selected_rounds=1),
            "short history": lambda a: a["round_history"].pop(),
            "round index": lambda a: a["round_history"][1].update(round=True),
            "broadcast": lambda a: a["round_history"][1].update(initial_model_fingerprint="b" * 64),
            "local seed": lambda a: a["round_history"][2]["sites"][0].update(loader_seed=0),
            "local model hash": lambda a: a["round_history"][2]["sites"][0].update(trained_model_fingerprint="bad"),
            "local count": lambda a: a["round_history"][2]["sites"][0].update(num_examples=6),
            "local updates": lambda a: a["round_history"][2]["sites"][0].update(gradient_updates=True),
            "weights": lambda a: a["round_history"][2].update(aggregation_weights=[0.4, 0.6]),
            "loss": lambda a: a["round_history"][2].update(validation_loss=0.0),
            "validation count": lambda a: a["round_history"][2]["validation_sites"][0].update(num_examples=2),
            "validation identity": lambda a: a["round_history"][2]["validation_sites"][0].update(site_idx=True),
            "split count": lambda a: a["site_splits"][0].update(validation_examples=2, fit_examples=4),
            "split seed": lambda a: a["site_splits"][0].update(split_seed=0),
            "optimization payload": lambda a: a["round_history"][2]["optimization_payload"].update(site_upload_bytes=0),
            "validation payload": lambda a: a["round_history"][2]["validation_payload"].update(site_upload_bytes=0),
            "communication": lambda a: a["communication"].update(total_synchronization_phases=200),
            "exposure": lambda a: a.update(training_examples_seen_per_site=[1200, 1200]),
            "timing component": lambda a: a["timing"].update(local_training_wall_seconds=0.0),
            "extra field": lambda a: a.update(evaluation={}),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                bad = deepcopy(self.result)
                mutation(bad["fit"])
                rehash_fit(bad["fit"])
                with self.assertRaises(ValueError):
                    self.validate(bad)

    def test_replayed_prior_stop_cannot_be_called_cap(self):
        bad = deepcopy(self.result)
        # Repair every loss summary and counter, but leave an earlier valid
        # patience stop in the middle of a superficially complete history.
        for index, item in enumerate(bad["fit"]["round_history"]):
            for site in item["validation_sites"]:
                site["loss_sum"] = float(site["num_examples"])
            item.update(validation_loss=1.0, stale_rounds=index)
        rehash_fit(bad["fit"])
        with self.assertRaisesRegex(ValueError, "already satisfied"):
            self.validate(bad)

    def test_unknown_recipe_and_nonfinite_evidence_fail_closed(self):
        for field, value in (("max_rounds", 1000), ("patience", 20), ("min_delta", 0.0),
                             ("refit_full_data", False), ("local_epochs", True), ("local_epochs", 99)):
            with self.subTest(field=field):
                config, bad = deepcopy(self.config), deepcopy(self.result)
                config["method"]["optimizer"][field] = value
                bad["fit"]["config"][field] = value
                bad["identity"]["config_sha256"] = digest(config)
                rehash_fit(bad["fit"])
                with self.assertRaises(ValueError):
                    self.validate(bad, config=config, identity=bad["identity"])
        for value in (float("nan"), float("inf"), -float("inf")):
            bad = deepcopy(self.result)
            bad["fit"]["round_history"][0]["validation_loss"] = value
            with self.assertRaises(ValueError):
                self.validate(bad)

    def test_outer_context_model_and_no_evaluation_are_strict(self):
        mutations = {
            "evaluation": lambda r: r.update(evaluation={}),
            "completion": lambda r: r.update(status="complete"),
            "outcome": lambda r: r.update(outcome="converged"),
            "seed bool": lambda r: r["identity"].update(seed=True),
            "site bool": lambda r: r["site_sizes"].__setitem__(0, True),
            "labels": lambda r: r["label_counts"]["0"].update({"1": 5, "0": 1}),
            "fingerprint": lambda r: r["fingerprints"].update(training="bad"),
            "model fingerprint": lambda r: r["fingerprints"].update(model="b" * 64),
            "unmeasured evaluation": lambda r: r["timing"]["stages"]["evaluation"].update(measured=True),
            "train failure": lambda r: r["timing"]["stages"]["train"].update(failed_calls=1),
            "short train timing": lambda r: r["timing"]["stages"]["train"].update(wall_seconds=0.0),
            "rss bool": lambda r: r["memory"].update(process_peak_rss_kib=True),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                bad = deepcopy(self.result)
                mutate(bad)
                with self.assertRaises(ValueError):
                    self.validate(bad)
        for invalid in (np.zeros((2,), dtype=np.float32), np.zeros((8, 5), dtype=np.float64),
                        np.full((8, 5), float("nan"), dtype=np.float32)):
            arrays = deepcopy(self.arrays)
            arrays["net.0.weight"] = invalid
            with self.assertRaises(ValueError):
                self.validate(arrays=arrays)

    def test_typed_error_keeps_diagnostic_context_and_never_evaluates(self):
        error = ScientificNoncompletion(deepcopy(self.result["fit"]), deepcopy(self.arrays))
        with patch.object(runner, "generate_datasets", return_value=tiny_datasets()), \
                patch.object(runner, "fit_method", side_effect=error), \
                patch.object(runner, "evaluate") as evaluate:
            with self.assertRaises(ScientificNoncompletion) as caught:
                runner.compute_cell(self.config, 25, "noniid")
        self.assertIs(caught.exception, error)
        self.assertEqual(digest(error.result["fit"]), digest(error.fit))
        self.assertEqual(error.result["fingerprints"], self.result["fingerprints"])
        self.assertEqual(error.result["label_counts"], self.result["label_counts"])
        self.assertNotIn("evaluation", error.result)
        evaluate.assert_not_called()

    def test_success_returns_the_original_model_and_audit_unchanged(self):
        config = self.config["method"]
        datasets = tiny_datasets()[0]
        sites = tensor_datasets_to_raw_sites(datasets, 2)

        def constant_validation(model, validation_sites):
            return 1.0, [dict(site_idx=site.site_idx, loss_sum=float(site.labels.numel()),
                              num_examples=int(site.labels.numel())) for site in validation_sites]

        with patch.object(fedavg, "_validation_loss", side_effect=constant_validation):
            expected_model, expected_audit = fedavg.fit_multiround_fedavg(
                sites, config["model"], fedavg.FedAvgConfig(**config["optimizer"]), 25)
            model, audit = runner.fit_method(sites, datasets, config, 25, MethodTimer("fixture"))
        self.assertEqual(fedavg._without_timings(audit), fedavg._without_timings(expected_audit))
        for key, value in model_arrays(model).items():
            np.testing.assert_array_equal(value, model_arrays(expected_model)[key])
        self.assertTrue(audit["completion_eligible"])
        self.assertEqual(audit["refit_rounds"], 1)

    def test_unknown_failure_is_not_converted_to_diagnostic_outcome(self):
        config = self.config["method"]
        datasets = tiny_datasets()[0]
        sites = tensor_datasets_to_raw_sites(datasets, 2)
        bad = deepcopy(self.result["fit"])
        bad["termination_reason"] = "unknown_failure"
        with patch.object(fedavg, "fit_multiround_fedavg", return_value=(object(), bad)):
            with self.assertRaises(RuntimeError) as caught:
                runner.fit_method(sites, datasets, config, 25, MethodTimer("fixture"))
        self.assertNotIsInstance(caught.exception, ScientificNoncompletion)
        with patch.object(fedavg, "fit_multiround_fedavg", side_effect=FloatingPointError("nonfinite loss")):
            with self.assertRaises(FloatingPointError):
                runner.fit_method(sites, datasets, config, 25, MethodTimer("fixture"))


if __name__ == "__main__":
    unittest.main()
