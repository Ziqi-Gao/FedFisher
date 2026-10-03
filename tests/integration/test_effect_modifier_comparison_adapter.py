"""Proposed comparison tasks exercise the same complete v2 boundary."""

import copy
import json
import unittest
from unittest import mock

import tests.integration.test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.causal_comparison import protocol


class ComparisonAdapterTests(unittest.TestCase):
    def test_proposed_running_fixtures_validate_from_disk(self):
        helper = fixture.FedFisherServerSchedulerAdapterTests()
        for name in ("comparison-pilot-running-job.json", "comparison-shard-running-job.json"):
            path, payload = helper.fixture(name)
            result = fixture.ADAPTER.load_running_job(path, helper.environment(path, payload))
            self.assertEqual(result["task"], payload["task"])

    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture("one-shot-running-job.json")
        self.job["resources"] = None
        self.job["requested_profile"] = None
        self.job["task"] = protocol.PILOT_TASK
        self.job["parameters"] = dict(variant="control", cell="r_unpenalized",
                                      shard_index=0, shard_count=1)

    def validate(self, job):
        return fixture.ADAPTER.validate_running_job(
            job, self.helper.environment(self.path, job), self.path)

    def test_every_proposed_cell_validates_only_with_strict_running_envelope(self):
        for task in protocol.TASKS:
            for cell in protocol.CELLS:
                job = copy.deepcopy(self.job)
                job["task"] = task
                job["parameters"]["cell"] = cell
                job["parameters"]["shard_count"] = 1 if task == protocol.PILOT_TASK else 16
                self.assertEqual(self.validate(job)["task"], task)
        job["allocation"]["gpu_count"] = 1
        with self.assertRaises(fixture.ADAPTER.AdapterError):
            self.validate(job)

    def test_new_task_names_dispatch_only_through_existing_foreground_adapter(self):
        for task in protocol.TASKS:
            self.job["task"] = task
            self.job["parameters"]["shard_count"] = 1 if task == protocol.PILOT_TASK else 16
            job = self.validate(self.job)
            with mock.patch.object(fixture.ADAPTER, "_safe_runtime") as runtime, mock.patch(
                    "effect_modifier_experiment.causal_comparison.runner.run_shard", return_value=0) as runner:
                self.assertEqual(fixture.ADAPTER.execute_task(job), 0)
                runtime.assert_called_once_with(job["job_id"], 1)
                runner.assert_called_once_with(job)

    def test_old_task_cannot_smuggle_new_cell_and_new_task_rejects_machine_choices(self):
        self.job["task"] = fixture.ADAPTER.ONE_SHOT_TASK
        with self.assertRaises(fixture.ADAPTER.AdapterError):
            self.validate(self.job)
        self.job["task"] = protocol.PILOT_TASK
        for key in ("resources", "gpu", "cpu_cores", "seed", "output_dir", "local_epochs"):
            job = copy.deepcopy(self.job)
            job["parameters"][key] = 1
            with self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)

    def test_schemas_examples_and_disabled_proposal_agree_with_fixed_registry(self):
        root = fixture.PROJECT_ROOT / "effect_modifier_experiment/server_scheduler"
        for stem, task in (("comparison-pilot", protocol.PILOT_TASK), ("comparison-shard", protocol.FORMAL_TASK)):
            schema = json.loads((root / (stem + "-parameters.schema.json")).read_text())
            example = json.loads((root / "examples" / (stem + ".job-request.json")).read_text())
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["properties"]["cell"]["enum"]), set(protocol.CELLS))
            self.assertEqual(example["task"], task)
            self.assertNotIn("resources", example)
            self.assertNotIn("execution_profile", example)
            protocol.validate_parameters(task, example["parameters"])
        proposal = (root / "comparison-registration-proposal.toml").read_text()
        self.assertIn("enabled = false", proposal)
        self.assertNotIn('kind = "gpu"', proposal)


if __name__ == "__main__":
    unittest.main()
