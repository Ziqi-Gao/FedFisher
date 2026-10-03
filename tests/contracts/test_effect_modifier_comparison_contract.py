"""Bounded, scratch-only protocol/artifact/evaluation integration fixtures."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from effect_modifier_experiment.causal_comparison import artifacts, evaluation, protocol
from effect_modifier_experiment.causal_comparison import computation as runner
from effect_modifier_experiment.causal_comparison.fedavg import FedAvgConfig, validate_fit_report as validate_fedavg_fit
from effect_modifier_experiment.causal_comparison.measurement import MethodTimer
from effect_modifier_experiment.causal_comparison.rlearner import LinearProbabilityEffectModel
from effect_modifier_experiment.dslassocov_benchmark import dataset_fingerprint, tensor_datasets_to_raw_sites, training_fingerprint
from effect_modifier_experiment.dslassocov_run_contract import canonical_text, sha256_file


def parameters(**changes):
    value = dict(variant="control", cell="r_unpenalized", shard_index=0, shard_count=1)
    value.update(changes)
    return value


def tiny_config(cell="r_unpenalized", variant="control"):
    config = protocol.run_config(protocol.PILOT_TASK, parameters(cell=cell, variant=variant))
    config["dgp"].update(covariate_dim=3, num_sites=2, num_test=8, site_min_samples=6, site_max_samples=8)
    config["evaluation"].update(repeats=2, batch_size=4)
    return config


def datasets(shift=0.0):
    def make(count, offset):
        raw = np.arange(count * 3, dtype=float).reshape(count, 3) / (count * 3) - 0.5 + offset
        treatment = (np.arange(count) % 2).astype(float)
        return TensorDataset(torch.tensor(np.column_stack((treatment, raw, treatment[:, None] * raw)), dtype=torch.float32),
                             torch.tensor(treatment, dtype=torch.long))
    return [make(6, shift), make(8, shift + 0.1)], make(8, 0.0), {0: {0: 3, 1: 3}, 1: {0: 4, 1: 4}}


def coherent_fake_provenance():
    values = dict(git_commit="1" * 40, git_dirty="0", git_status_sha256=hashlib.sha256(b"").hexdigest(), source_tree_sha256="2" * 64)
    values["provenance_sha256"] = hashlib.sha256(canonical_text(values, artifacts.PROVENANCE_KEYS).encode()).hexdigest()
    return values


class ComparisonProtocolTests(unittest.TestCase):
    def test_fixed_scientific_cells_and_shards_have_exact_disjoint_coverage(self):
        self.assertEqual(len(protocol.CELLS), 24)
        self.assertTrue({"dslassocov", "r_unpenalized", "r_adaptive_lasso"}.issubset(protocol.CELLS))
        for count in (8, 16):
            sequence = [protocol.seed_sequence(protocol.FORMAL_TASK, parameters(shard_count=count, shard_index=index)) for index in range(count)]
            combined = [seed for shard in sequence for seed in shard]
            self.assertEqual(sorted(combined), list(range(300)))
            self.assertEqual(len(set(combined)), 300)
            for index, shard in enumerate(sequence):
                self.assertTrue(all(seed % count == index for seed in shard))
        self.assertEqual(protocol.seed_sequence(protocol.PILOT_TASK, parameters()), (0,))
        self.assertNotEqual(protocol.run_id(protocol.PILOT_TASK, parameters()), protocol.run_id(protocol.FORMAL_TASK, parameters(shard_count=8)))

    def test_requests_reject_hardware_or_arbitrary_work_and_boolean_shards(self):
        for changes in (
            {"cpu_cores": 4}, {"gpu_count": 1}, {"seed": 91}, {"output_dir": "/data/arbitrary"},
            {"cell": "python_script"}, {"cell": []}, {"variant": "new_dgp"},
            {"shard_index": True}, {"shard_count": True}, {"shard_index": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.PILOT_TASK, parameters(**changes))
        with self.assertRaises(ValueError):
            protocol.validate_parameters("old-task", parameters())


class ComparisonEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_zero_hundred_variable_ties_do_not_reward_signal_index(self):
        result = evaluation.signal_recovery_with_ties(np.zeros(100), 2, "tau_intervention_prob")
        self.assertEqual(result["expected_topk_hits"], 0.04)
        self.assertEqual(result["expected_topk_precision"], 0.02)
        self.assertEqual(result["probability_all_signals_selected"], 1 / 4950)
        self.assertEqual(result["auroc"], 0.5)
        self.assertEqual(result["deterministic_index_tiebreak_diagnostic"]["topk_hits"], 2)

    def test_unique_and_partially_tied_selection_boundaries(self):
        unique = evaluation.signal_recovery_with_ties([4, 3, 2, 1], 2, "tau_intervention_prob")
        self.assertEqual(unique["expected_topk_hits"], 2)
        self.assertEqual(unique["probability_all_signals_selected"], 1)
        tied = evaluation.signal_recovery_with_ties([4, 2, 2, 2], 2, "tau_intervention_prob")
        self.assertAlmostEqual(tied["expected_topk_hits"], 4 / 3)
        self.assertAlmostEqual(tied["probability_all_signals_selected"], 1 / 3)
        excluded = evaluation.signal_recovery_with_ties([3, 0, 2, 1], 2, "tau_intervention_prob")
        self.assertEqual(excluded["expected_topk_hits"], 1)
        self.assertEqual(excluded["probability_all_signals_selected"], 0)

    def test_effect_only_has_probability_scale_and_shared_permutations(self):
        sites, test, _ = datasets()
        config = tiny_config()
        first = LinearProbabilityEffectModel(np.array([0.0, 0.8, -0.4, 0.0]))
        second = LinearProbabilityEffectModel(np.array([4.0, 1.6, -0.8, 0.0]))
        results = [evaluation.evaluate(model, sites, test, variant="control", split="iid", seed=7,
                    dgp=config["dgp"], config=config["evaluation"]) for model in (first, second)]
        self.assertEqual(results[0]["permutation_sha256"], results[1]["permutation_sha256"])
        self.assertEqual(set(results[0]["metrics"]), {"permute/tau_prob", "zero/tau_prob"})
        self.assertTrue(results[0]["effect_only"])
        self.assertNotIn("accuracy", results[0])
        np.testing.assert_allclose(results[1]["metrics"]["permute/tau_prob"]["importance"],
                                   2 * np.asarray(results[0]["metrics"]["permute/tau_prob"]["importance"]), atol=1e-14)

    def test_outcome_evaluation_always_reconstructs_consistent_counterfactual_inputs(self):
        observed = []

        class CheckedOutcome(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.placeholder = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, inputs):
                observed.append(inputs.detach().clone())
                treatment, raw, interaction = inputs[:, :1], inputs[:, 1:4], inputs[:, 4:]
                if not torch.equal(interaction, treatment * raw):
                    raise AssertionError("counterfactual interaction inputs are inconsistent")
                score = treatment[:, 0] * raw[:, 0] + self.placeholder
                return torch.stack((torch.zeros_like(score), score), dim=1)

        sites, test, _ = datasets()
        config = tiny_config()
        result = evaluation.evaluate(CheckedOutcome(), sites, test, variant="control", split="iid", seed=7,
                                     dgp=config["dgp"], config=config["evaluation"])
        self.assertFalse(result["effect_only"])
        self.assertEqual(set(result["metrics"]), {"permute/tau_prob", "zero/tau_prob", "permute/tau_logit", "zero/tau_logit"})
        self.assertTrue(observed)
        self.assertTrue(any(torch.all(batch[:, 0] == 0) for batch in observed))
        self.assertTrue(any(torch.all(batch[:, 0] == 1) for batch in observed))
        effect = evaluation.evaluate(LinearProbabilityEffectModel(np.array([0., 1., 0., 0.])), sites, test,
                                     variant="control", split="iid", seed=7, dgp=config["dgp"], config=config["evaluation"])
        self.assertEqual(result["permutation_sha256"], effect["permutation_sha256"])

    def test_local_interventions_are_sample_weighted_and_exhaustive(self):
        sites, test, _ = datasets()
        config = tiny_config(variant="local_permutation")
        model = LinearProbabilityEffectModel(np.array([0., 1., 0., 0.]))
        result = evaluation.evaluate(model, sites, test, variant="local_permutation", split="iid", seed=4,
                                    dgp=config["dgp"], config=config["evaluation"])
        raw = test.tensors[0][:, 1].numpy()
        self.assertAlmostEqual(result["metrics"]["zero/tau_prob"]["importance"][0], float(np.abs(raw).mean()), places=7)
        self.assertEqual(result["shard_sizes"], [3, 5])
        self.assertEqual(sum(result["shard_sizes"]), len(test))
        bad_config = deepcopy(config["evaluation"])
        bad_config["modes"] = ["unknown"]
        with self.assertRaises(ValueError):
            evaluation.evaluate(model, sites, test, variant="control", split="iid", seed=4, dgp=config["dgp"], config=bad_config)


class ComparisonArtifactTests(unittest.TestCase):
    def setUp(self):
        parent = artifacts.safe_path(Path("/scr/del6500/FedFisher/scheduler-regression/comparison-contract-fixtures"))
        parent.mkdir(parents=True, exist_ok=True)
        artifacts.safe_path(parent)
        self.temporary = tempfile.TemporaryDirectory(prefix="cell-", dir=str(parent))
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        # These tests isolate the store's contract. R's numerical report
        # validator has its own dedicated tests; runner composition is below.
        self.report_validator = patch("effect_modifier_experiment.causal_comparison.rlearner.validate_fit_report")
        self.mock_report_validator = self.report_validator.start()
        self.addCleanup(self.report_validator.stop)
        self.config = tiny_config()
        name = protocol.run_id(protocol.PILOT_TASK, parameters())
        self.manifest = artifacts.make_manifest(name, self.config, coherent_fake_provenance())
        self.store = artifacts.ArtifactStore(self.directory / name, self.manifest)
        self.store.initialize()

    def _result(self):
        sites, test, labels = datasets()
        model = LinearProbabilityEffectModel(np.array([0.1, 0.8, -0.4, 0.0]))
        arrays = evaluation.model_arrays(model)
        timer = MethodTimer(self.config["cell"])
        with timer.measure("train"):
            pass
        with timer.measure("evaluation"):
            measured = evaluation.evaluate(model, sites, test, variant="control", split="iid", seed=0,
                                           dgp=self.config["dgp"], config=self.config["evaluation"])
        return dict(identity=self.store.identity(0, "iid"), status="complete",
                    fit={"completion_eligible": True, "fit_fingerprint": "3" * 64, "seed": 0,
                         "num_sites": 2, "num_examples": 14,
                         "local_reports": [{"site_idx": index, "summary": {"num_examples": count}} for index, count in enumerate((6, 8))]},
                    evaluation=measured, timing=timer.to_dict(), site_sizes=[6, 8],
                    label_counts={str(site): {str(label): count for label, count in counts.items()} for site, counts in labels.items()},
                    fingerprints=dict(training=training_fingerprint(tensor_datasets_to_raw_sites(sites, 3)),
                                      test=dataset_fingerprint(test), model=evaluation.model_hash(arrays))), arrays

    def _complete(self):
        self.store.start(0, "iid", {"job_id": "fixture"})
        result, arrays = self._result()
        self.store.complete(0, "iid", result, arrays)
        return result, arrays

    def test_missing_start_complete_retry_is_immutable(self):
        self.assertEqual(self.store.status(0, "iid"), "missing")
        self._complete()
        path = self.store.cell_dir(0, "iid")
        before = {item.name: sha256_file(item) for item in path.iterdir()}
        retry = artifacts.ArtifactStore(self.store.directory, self.manifest)
        retry.initialize()
        self.assertEqual(retry.status(0, "iid"), "validated")
        with self.assertRaises(FileExistsError):
            retry.start(0, "iid", {"job_id": "retry"})
        self.assertEqual(before, {item.name: sha256_file(item) for item in path.iterdir()})
        self.assertTrue(self.mock_report_validator.called)

    def test_partial_and_failed_attempts_are_never_resumed_or_overwritten(self):
        self.store.start(0, "iid", {"job_id": "fixture"})
        with self.assertRaisesRegex(ValueError, "partial"):
            self.store.status(0, "iid")
        self.store.fail(0, "iid", RuntimeError("bounded fixture failure"))
        with self.assertRaisesRegex(ValueError, "partial"):
            self.store.start(0, "iid", {"job_id": "retry"})
        self.assertEqual(set(path.name for path in self.store.cell_dir(0, "iid").iterdir()), {"started.json", "failure.json"})

    def test_ineligible_or_semantically_wrong_results_cannot_create_done_marker(self):
        self.store.start(0, "iid", {"job_id": "fixture"})
        original, arrays = self._result()
        cases = []
        bad = deepcopy(original); bad["fit"]["completion_eligible"] = False; cases.append(bad)
        bad = deepcopy(original); bad["identity"]["seed"] = 1; cases.append(bad)
        bad = deepcopy(original); bad["evaluation"]["metrics"]["permute/tau_prob"]["recovery"]["expected_topk_hits"] = -1; cases.append(bad)
        bad = deepcopy(original); bad["evaluation"]["metrics"]["permute/tau_logit"] = deepcopy(bad["evaluation"]["metrics"]["permute/tau_prob"]); cases.append(bad)
        bad = deepcopy(original); bad["timing"]["stages"]["train"]["wall_seconds"] = -1; cases.append(bad)
        bad = deepcopy(original); bad["evaluation"]["metrics"]["permute/tau_prob"]["importance"][0] = float("nan"); cases.append(bad)
        for result in cases:
            with self.assertRaises(ValueError):
                self.store.complete(0, "iid", result, arrays)
            self.assertEqual(set(path.name for path in self.store.cell_dir(0, "iid").iterdir()), {"started.json"})
        wrong_arrays = {"effect_coefficients": np.ones(3, dtype=np.float64)}
        with self.assertRaises(ValueError):
            self.store.complete(0, "iid", original, wrong_arrays)

    def test_rehashed_tampered_semantics_still_fail(self):
        self._complete()
        cell = self.store.cell_dir(0, "iid")
        result = artifacts.read_json(cell / "result.json")
        result["evaluation"]["metrics"]["permute/tau_prob"]["recovery"]["expected_topk_hits"] = 0.123
        (cell / "result.json").write_text(artifacts.canonical(result))
        marker = artifacts.read_json(cell / "done.json")
        marker["artifacts"]["result.json"] = sha256_file(cell / "result.json")
        (cell / "done.json").write_text(artifacts.canonical(marker))
        with self.assertRaisesRegex(ValueError, "recovery"):
            self.store.status(0, "iid")

    def test_hash_identity_manifest_and_symlinks_fail_closed(self):
        self._complete()
        cell = self.store.cell_dir(0, "iid")
        with (cell / "model.npz").open("ab") as stream:
            stream.write(b"fixture checksum mutation")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store.status(0, "iid")
        wrong = dict(self.manifest, provenance_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "provenance"):
            artifacts.ArtifactStore(self.store.directory, wrong)
        alias = self.directory / "alias"
        alias.symlink_to(self.store.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            artifacts.safe_path(alias / "run_manifest.env")
        with self.assertRaises(ValueError):
            artifacts.safe_path(Path("relative/path"))
        duplicate = self.directory / "duplicate.json"
        duplicate.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            artifacts.read_json(duplicate)

    def test_current_provenance_requires_clean_source(self):
        with patch.object(artifacts, "compute_provenance", return_value=coherent_fake_provenance()):
            self.assertEqual(artifacts.current_provenance()["git_dirty"], "0")
        with patch.object(artifacts, "compute_provenance", return_value=dict(coherent_fake_provenance(), git_dirty="1")):
            with self.assertRaisesRegex(ValueError, "clean"):
                artifacts.current_provenance()


class ComparisonRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tiny_real_fedavg_dispatch_and_semantic_audit(self):
        sites, _, _ = datasets()
        raw = tensor_datasets_to_raw_sites(sites, 3)
        method = dict(family="fedavg", model="SyntheticMLP", optimizer=asdict(FedAvgConfig(
            covariate_dim=3, batch_size=4, min_rounds=1, max_rounds=1,
            stopping_mode="fixed_total_epochs", total_local_epochs=1)))
        timer = MethodTimer("fixture")
        model, audit = runner.fit_method(raw, sites, method, 6, timer)
        arrays = evaluation.model_arrays(model)
        validate_fedavg_fit(audit, method["optimizer"], method["model"], 6, arrays)
        self.assertTrue(timer.to_dict()["stages"]["train"]["measured"])
        for field, value in (("total_gradient_updates", -1), ("safety_limit_reached", True), ("fit_fingerprint", "0" * 64)):
            invalid = deepcopy(audit); invalid[field] = value
            with self.assertRaises(ValueError):
                validate_fedavg_fit(invalid, method["optimizer"], method["model"], 6, arrays)

    def test_r_dispatch_rejects_nonconvergence_and_legacy_boundary_is_explicit(self):
        sites, _, _ = datasets()
        raw = tensor_datasets_to_raw_sites(sites, 3)
        method = tiny_config()["method"]
        model = LinearProbabilityEffectModel(np.array([0., 1., 0., 0.]))
        with patch("effect_modifier_experiment.causal_comparison.rlearner.fit_homogeneous_rlearner", return_value=(model, {"converged": False})) as fit:
            with self.assertRaisesRegex(RuntimeError, "completion rule"):
                runner.fit_method(raw, sites, method, 8, MethodTimer("fixture"))
            self.assertEqual(fit.call_args.kwargs["covariate_dim"], 3)
            self.assertNotIn("test_dataset", fit.call_args.kwargs)
        with patch("effect_modifier_experiment.causal_comparison.legacy.fit_legacy", return_value=(model, {"completion_eligible": True})) as fit:
            runner.fit_method(raw, sites, {"family": "dslassocov"}, 8, MethodTimer("fixture"))
            self.assertIs(fit.call_args.args[0], raw)

    def test_four_variants_preserve_paired_training_and_common_test_and_fit(self):
        def generation(args):
            return datasets(shift=0.4 if "gaussian" in args.variant else 0.0)

        def fitting(raw_sites, site_datasets, method, seed, timer):
            with timer.measure("train"):
                scale = float(raw_sites[0].features[:, 1].mean())
                model = LinearProbabilityEffectModel(np.array([0., 1. + scale, -0.5, 0.]))
            return model, {"completion_eligible": True, "fit_fingerprint": training_fingerprint(raw_sites)}

        records = {}
        with patch.object(runner, "generate_datasets", side_effect=generation), patch.object(runner, "fit_method", side_effect=fitting):
            for variant in protocol.VARIANTS:
                result, _ = runner.compute_cell(tiny_config(variant=variant), 0, "iid")
                records[variant] = result
        from effect_modifier_experiment.causal_comparison.reporting import validate_pairs
        validate_pairs(records)
        self.assertEqual(len({result["fingerprints"]["test"] for result in records.values()}), 1)
        self.assertEqual(records["control"]["fingerprints"]["training"], records["local_permutation"]["fingerprints"]["training"])
        self.assertNotEqual(records["control"]["fingerprints"]["training"], records["site_gaussian_shift"]["fingerprints"]["training"])
        broken = deepcopy(records)
        broken["local_permutation"]["fit"]["fit_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fit"):
            validate_pairs(broken)


if __name__ == "__main__":
    unittest.main()
