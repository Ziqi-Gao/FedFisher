"""Tiny real-fit composition tests; no default DGP, pilot, or central execution."""

import copy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from effect_modifier_experiment.dslassocov_run_contract import canonical_text
from effect_modifier_experiment.one_shot_effects import (
    artifacts, dr_sieve, fedfisher_em, fisher_summary, model_io, protocol, runner, validation,
)
from effect_modifier_experiment.one_shot_effects.evaluation import score_agreement
from effect_modifier_experiment.one_shot_effects.wire import canonical, unpad_payload, unpack_payload


SCRATCH = Path("/scr/del6500/FedFisher/one-shot-effects-validation")
RECIPES = ("dr_linear", "fedavg_shallow", "linearity_shallow")
COMPRESSION_RECIPE = "compression_shallow_fisher_gradient_r16"


def tiny_config(recipe):
    config = protocol.run_config(protocol.PILOT_TASK, dict(recipe=recipe, dataset="scmv2_base"))
    n = 120 if recipe == "dr_linear" else 20
    config["dgp"].update(num_sites=2, site_min_samples=n, site_max_samples=n,
                         covariate_dim=5, num_test=40)
    config["fisher"].update(local_epochs=1, batch_size=8, basis_max_rows=2, range_oversampling=2)
    config["legacy_diagnostic"].update(iterations=2, evaluate_every=1)
    config["dr_nuisance"].update(crossfit_folds=5, nuisance_cv_folds=2,
                                  nuisance_lambda_count=2, nuisance_lambda_min_ratio=.1)
    config["dr_sieve"].update(lambda_count=3, lambda_min_ratio=.1)
    return config


def fixture_provenance():
    """Synthetic immutable source binding; real provenance is tested separately."""
    value = dict(git_commit="a" * 40, git_dirty="0",
                 git_status_sha256=hashlib.sha256(b"").hexdigest(),
                 source_tree_sha256=hashlib.sha256(b"runner-tiny-fixture").hexdigest())
    value["provenance_sha256"] = hashlib.sha256(
        canonical_text(value, artifacts.PROVENANCE_KEYS).encode()
    ).hexdigest()
    return value


class OneShotEffectsRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cases = {}
        cls.provenance = fixture_provenance()
        cls.probe = np.linspace(-.8, .9, 20).reshape(4, 5)

    def setUp(self):
        artifacts.safe_path(SCRATCH)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        artifacts.safe_path(SCRATCH)
        self.temporary = tempfile.TemporaryDirectory(prefix="runner-integration-", dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    @classmethod
    def case(cls, recipe):
        if recipe not in cls.cases:
            config = tiny_config(recipe)
            predictions = {}

            def archive(model, name, arrays):
                predictions[name] = model.predict_tau(cls.probe)
                return model_io.archive_model(model, name, arrays)

            with ExitStack() as stack:
                # Stage 2 must not construct a sketch; the compression fixture
                # exercises the actual sketch, wire and server implementations.
                if config["recipe_definition"]["family"] != "compression":
                    for module in (fedfisher_em, fisher_summary):
                        stack.enter_context(mock.patch.object(module, "build_basis", side_effect=AssertionError(
                            "stage 2 must not build a compression basis")))
                stack.enter_context(mock.patch.object(runner, "archive_model", side_effect=archive))
                result, arrays, payloads = runner.compute_cell(config)
            identity = dict(run_id=config["run_id"], seed=0, recipe=recipe,
                            dataset=config["dataset"],
                            config_sha256=hashlib.sha256(canonical(config).encode()).hexdigest(),
                            provenance_sha256=cls.provenance["provenance_sha256"])
            result.update(identity=identity, prerequisite=None)
            cls.cases[recipe] = (config, result, arrays, payloads, predictions)
        return copy.deepcopy(cls.cases[recipe])

    def validate(self, config, result, arrays, payloads):
        # Only this fixture relaxes production's fixed config comparison.
        # The real semantic validator and all numerical checks remain active.
        with mock.patch.object(protocol, "run_config", return_value=config):
            validation.validate_result(result, arrays, payloads, config, result["identity"])

    def store(self, config):
        return artifacts.EffectArtifactStore(
            config, self.provenance, self.directory / config["run_id"]
        )

    def test_actual_dr_fedavg_and_full_gradient_cells_validate_without_compression(self):
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                config, result, arrays, payloads, _ = self.case(recipe)
                self.validate(config, result, arrays, payloads)
                self.assertEqual(len(payloads), 2)
                self.assertEqual(result["data_fingerprints"]["test_rows"], 40)
                self.assertEqual(result["communication"]["upload_rounds"], 1)
                self.assertEqual(result["communication"]["upload_bytes"], sum(map(len, payloads)))
                self.assertIsNone(result["communication"]["wire_budget_per_site"])
                self.assertEqual(result["communication"]["padding_bytes"], 0)
                if recipe == "dr_linear":
                    for audit in result["local_audits"]:
                        self.assertTrue(audit["nuisance"]["converged"])
                        self.assertEqual(len(audit["nuisance"]["crossfit_folds"]), 5)
                        self.assertEqual(audit["nuisance"]["config"]["nuisance_lambda_count"], 2)
                        self.assertEqual(audit["counts"], dict(T=84, V=18, H=18))
                    holdout = result["models"]["primary"]["holdout"]
                    np.testing.assert_allclose(holdout["summary_squared_scores"],
                                               holdout["exact_squared_scores"], atol=1e-10)
                elif recipe == "fedavg_shallow":
                    self.assertFalse(result["models"]["primary"]["fit"]["fisher_computed"])
                else:
                    self.assertEqual(len(result["diagnostics"]["local_linearization"]), 2)
                    self.assertIs(type(result["diagnostics"]["linearization"]["gate"][
                        "compression_eligible"]), bool)

    def test_archived_models_restore_identical_probability_effect_predictions(self):
        for recipe in RECIPES:
            _, result, arrays, _, predictions = self.case(recipe)
            for name, record in result["models"].items():
                with self.subTest(recipe=recipe, model=name):
                    restored = model_io.restore_model(record, arrays)
                    np.testing.assert_array_equal(restored.predict_tau(self.probe), predictions[name])
                    altered = {key: value.copy() for key, value in arrays.items()}
                    altered[next(iter(record["array_keys"].values()))].flat[0] += .125
                    with self.assertRaisesRegex(ValueError, "fingerprint"):
                        model_io.restore_model(record, altered)

    def test_production_validator_rejects_tiny_or_caller_changed_configuration(self):
        config, result, arrays, payloads, _ = self.case("fedavg_shallow")
        with self.assertRaisesRegex(ValueError, "reviewed fixed pilot recipe"):
            validation.validate_result(result, arrays, payloads, config, result["identity"])
        changed = copy.deepcopy(config)
        changed["seed"] = 1
        with mock.patch.object(protocol, "run_config", return_value=config), self.assertRaisesRegex(
                ValueError, "reviewed fixed pilot recipe"):
            validation.validate_result(result, arrays, payloads, changed, result["identity"])

    def test_consistently_recomputed_summary_comparison_cannot_hide_changed_scores(self):
        config, result, arrays, payloads, _ = self.case("dr_linear")
        holdout = result["models"]["primary"]["holdout"]
        holdout["summary_squared_scores"][0] += .5
        holdout["summary_comparison"] = score_agreement(
            holdout["exact_squared_scores"], holdout["summary_squared_scores"]
        )
        with self.assertRaises(ValueError):
            self.validate(config, result, arrays, payloads)

    def test_changed_full_gradient_gate_is_not_accepted_as_saved_diagnostics(self):
        config, result, arrays, payloads, _ = self.case("linearity_shallow")
        gate = result["diagnostics"]["linearization"]["gate"]
        gate["compression_eligible"] = not gate["compression_eligible"]
        with self.assertRaisesRegex(ValueError, "linearization evidence or gate"):
            self.validate(config, result, arrays, payloads)

    def test_compression_missing_or_failed_prerequisite_never_starts_compute(self):
        config = tiny_config(COMPRESSION_RECIPE)
        prerequisite_config = tiny_config("linearity_shallow")
        job = dict(task=protocol.PILOT_TASK,
                   parameters=dict(recipe=config["recipe"], dataset=config["dataset"]))

        def fixed_config(task, parameters):
            self.assertEqual(task, protocol.PILOT_TASK)
            return config if parameters["recipe"] == COMPRESSION_RECIPE else prerequisite_config

        for state in ("missing", "failed_gate", "partial"):
            with self.subTest(state=state):
                store = mock.Mock()
                store.unit_dir = self.directory / "prerequisite_not_created"
                store.status.return_value = "missing" if state == "missing" else "validated"
                if state == "partial":
                    store.status.side_effect = ValueError("partial prerequisite artifact")
                gate_result = dict(diagnostics=dict(linearization=dict(gate=dict(compression_eligible=False))))
                with mock.patch.object(protocol, "run_config", side_effect=fixed_config), mock.patch.object(
                        runner, "current_provenance", return_value=self.provenance), mock.patch.object(
                        runner, "EffectArtifactStore", return_value=store) as make_store, mock.patch.object(
                        runner, "read_json", return_value=gate_result) as read_result, mock.patch.object(
                        runner, "compute_cell") as compute:
                    with self.assertRaises(ValueError):
                        runner.run_job(job)
                    make_store.assert_called_once_with(prerequisite_config, self.provenance)
                    compute.assert_not_called()
                    store.initialize.assert_not_called()
                    store.start.assert_not_called()
                    if state != "failed_gate":
                        read_result.assert_not_called()
                self.assertFalse(store.unit_dir.exists())

    def test_real_compression_padded_uploads_server_summary_and_model_archive(self):
        config, result, arrays, payloads, predictions = self.case(COMPRESSION_RECIPE)
        prerequisite_config = tiny_config("linearity_shallow")
        # This synthetic accepted prerequisite tests the admission binding;
        # it does not claim that any scientific linearity pilot passed.
        prerequisite = dict(
            identity=dict(run_id=prerequisite_config["run_id"], seed=0,
                          recipe="linearity_shallow", dataset=config["dataset"],
                          config_sha256=hashlib.sha256(canonical(prerequisite_config).encode()).hexdigest(),
                          provenance_sha256=self.provenance["provenance_sha256"]),
            result_sha256=hashlib.sha256(b"accepted-linearity-fixture-only").hexdigest(),
            gate=dict(compression_eligible=True),
        )
        result["prerequisite"] = prerequisite

        def fixed_config(task, parameters):
            self.assertEqual(task, protocol.PILOT_TASK)
            return config if parameters["recipe"] == COMPRESSION_RECIPE else prerequisite_config

        with mock.patch.object(protocol, "run_config", side_effect=fixed_config), mock.patch.object(
                artifacts, "current_provenance", return_value=self.provenance), mock.patch.object(
                runner, "compression_prerequisite", return_value=prerequisite) as accepted:
            validation.validate_result(result, arrays, payloads, config, result["identity"])
            accepted.assert_called_once_with(config, self.provenance)
            store = self.store(config)
            store.initialize()
            store.start(dict(job_id="compression-unit-fixture", execution_profile="fixture"))
            store.complete(result, arrays, payloads)
            self.assertEqual(store.status(), "validated")
        budget = protocol.compression_wire_budget(config)
        self.assertEqual([len(blob) for blob in payloads], [budget, budget])
        communication = result["communication"]
        self.assertEqual(communication["upload_bytes"], 2 * budget)
        self.assertEqual(communication["native_upload_bytes"], sum(len(unpad_payload(blob)) for blob in payloads))
        self.assertEqual(communication["padding_bytes"], 2 * budget - communication["native_upload_bytes"])
        self.assertGreater(communication["padding_bytes"], 0)
        self.assertEqual(communication["wire_budget_per_site"], budget)
        for blob in payloads:
            decoded, _ = unpack_payload(blob)
            self.assertEqual(decoded["basis"].shape[1], 16)
        primary = result["models"]["primary"]
        restored = model_io.restore_model(primary, arrays)
        np.testing.assert_array_equal(restored.predict_tau(self.probe), predictions["primary"])
        np.testing.assert_allclose(primary["holdout"]["summary_squared_scores"],
                                   result["diagnostics"]["linearization"]["scores"]["low_rank"], atol=1e-12)

    def test_tiny_aggregation_replays_validation_rule_and_restores_all_models(self):
        config, result, arrays, payloads, predictions = self.case("aggregation_shallow")
        self.validate(config, result, arrays, payloads)
        diagnostic = result["diagnostics"]["legacy_rule"]
        self.assertEqual(diagnostic["status"], "complete")
        self.assertEqual(diagnostic["report"]["legacy_configuration"]["iterations"], 2)
        self.assertEqual(diagnostic["report"]["legacy_configuration"]["evaluate_every"], 1)
        self.assertEqual(set(result["models"]), {
            "primary", "matched_fedavg", "legacy_validation_selected", "strict_legacy_uploaded_objective"
        })
        for name, record in result["models"].items():
            np.testing.assert_array_equal(model_io.restore_model(record, arrays).predict_tau(self.probe), predictions[name])

    def test_real_artifact_store_semantics_reject_rehashed_corruption_without_inference(self):
        config, result, arrays, payloads, _ = self.case("dr_linear")
        with mock.patch.object(artifacts, "current_provenance", return_value=self.provenance), mock.patch.object(
                protocol, "run_config", return_value=config):
            store = self.store(config)
            self.assertEqual(result["identity"], store.identity)
            store.initialize()
            store.start(dict(job_id="runner-fixture", execution_profile="fixture"))
            store.complete(result, arrays, payloads)
            with mock.patch.object(runner, "generate_data", side_effect=AssertionError("status cannot generate")), mock.patch.object(
                    dr_sieve, "fit_dr_pseudo_outcomes", side_effect=AssertionError("status cannot fit")), mock.patch.object(
                    dr_sieve.DRSieveModel, "predict_tau", side_effect=AssertionError("status cannot infer")):
                self.assertEqual(store.status(), "validated")
                result_path = store.unit_dir / "result.json"
                changed = json.loads(result_path.read_text())
                holdout = changed["models"]["primary"]["holdout"]
                holdout["summary_squared_scores"][0] += .5
                holdout["summary_comparison"] = score_agreement(
                    holdout["exact_squared_scores"], holdout["summary_squared_scores"]
                )
                result_path.write_text(canonical(changed))
                marker_path = store.unit_dir / "done.json"
                marker = json.loads(marker_path.read_text())
                marker["artifacts"]["result.json"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
                marker_path.write_text(canonical(marker))
                with self.assertRaises(ValueError):
                    store.status()

    def test_run_job_publishes_identity_and_revalidates_before_skip(self):
        config, result, arrays, payloads, _ = self.case("fedavg_shallow")
        job = dict(task=protocol.PILOT_TASK,
                   parameters=dict(recipe=config["recipe"], dataset=config["dataset"]),
                   job_id="tiny-runner-fixture", execution_profile="fixture",
                   allocation=dict(cpu_cores=1, memory_mib=8192), estimated_runtime_seconds=1)
        with mock.patch.object(artifacts, "current_provenance", return_value=self.provenance), mock.patch.object(
                runner, "current_provenance", return_value=self.provenance), mock.patch.object(
                protocol, "run_config", return_value=config), mock.patch.dict(
                os.environ, {"SERVER_SCHEDULER_ATTEMPT": "1"}):
            store = self.store(config)
            result.pop("identity")
            with mock.patch.object(runner, "EffectArtifactStore", return_value=store), mock.patch.object(
                    runner, "compute_cell", return_value=(result, arrays, payloads)) as compute:
                self.assertEqual(runner.run_job(job), 0)
                self.assertEqual(store.status(), "validated")
                compute.assert_called_once_with(config)
                self.assertEqual(runner.run_job(job), 0)
                compute.assert_called_once_with(config)
            saved = json.loads((store.unit_dir / "result.json").read_text())
            self.assertEqual(saved["identity"], store.identity)

    def test_run_job_failure_leaves_partial_evidence_and_cannot_retry_overwrite(self):
        config = tiny_config("fedavg_shallow")
        job = dict(task=protocol.PILOT_TASK,
                   parameters=dict(recipe=config["recipe"], dataset=config["dataset"]),
                   job_id="tiny-failed-runner-fixture", execution_profile="fixture",
                   allocation=dict(cpu_cores=1, memory_mib=8192), estimated_runtime_seconds=1)
        with mock.patch.object(artifacts, "current_provenance", return_value=self.provenance), mock.patch.object(
                runner, "current_provenance", return_value=self.provenance), mock.patch.object(
                protocol, "run_config", return_value=config), mock.patch.dict(
                os.environ, {"SERVER_SCHEDULER_ATTEMPT": "1"}):
            store = self.store(config)
            with mock.patch.object(runner, "EffectArtifactStore", return_value=store), mock.patch.object(
                    runner, "compute_cell", side_effect=RuntimeError("fixture numerical failure")) as compute:
                with self.assertRaisesRegex(RuntimeError, "fixture numerical failure"):
                    runner.run_job(job)
                failure_path = store.unit_dir / "failure.json"
                before = failure_path.read_bytes()
                self.assertEqual(json.loads(before)["identity"], store.identity)
                self.assertFalse((store.unit_dir / "done.json").exists())
                with self.assertRaises(ValueError):
                    runner.run_job(job)
                compute.assert_called_once_with(config)
                self.assertEqual(failure_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
