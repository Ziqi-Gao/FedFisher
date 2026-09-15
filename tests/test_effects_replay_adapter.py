"""No-fit fixtures for the proposed fixed-seed checkpoint replay task."""

import copy
import json
import re
import sys
from types import ModuleType
import unittest
from unittest import mock

import test_fedfisher_server_scheduler_adapter as fixture
from effect_modifier_experiment.one_shot_effects.diagnostics import protocol


PROPOSAL_ROOT = (
    fixture.PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/effects-replay-v1"
)


class EffectsReplayAdapterTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixture.FedFisherServerSchedulerAdapterTests()
        self.path, self.job = self.helper.fixture("effects-replay-pilot-running-job.json")

    def validate(self, job, environment=None):
        return fixture.ADAPTER.validate_running_job(
            job, self.helper.environment(self.path, job) if environment is None else environment,
            self.path,
        )

    def test_running_fixture_loads_and_dispatches_through_the_only_foreground_entrypoint(self):
        job = fixture.ADAPTER.load_running_job(
            self.path, self.helper.environment(self.path, self.job)
        )
        self.assertEqual(job["task"], protocol.PILOT_TASK)
        self.assertEqual(job["allocation"]["memory_mib"], 8192)
        self.assertIsNone(job["resources"])
        runner = ModuleType("effect_modifier_experiment.one_shot_effects.diagnostics.runner")
        runner.run_job = mock.Mock(return_value=23)
        with mock.patch.dict(sys.modules, {runner.__name__: runner}), mock.patch.object(
            fixture.ADAPTER, "_safe_runtime"
        ) as runtime:
            self.assertEqual(fixture.ADAPTER.run_from_manifest(
                self.path, self.helper.environment(self.path, self.job)
            ), 23)
        runtime.assert_called_once_with(job["job_id"], 1)
        runner.run_job.assert_called_once_with(job)

    def test_exactly_five_cases_use_the_same_fixed_seed_replay_contract(self):
        self.assertEqual(len(protocol.cohort_plan()), 5)
        for parameters in protocol.cohort_plan():
            job = copy.deepcopy(self.job)
            job["parameters"] = parameters
            with self.subTest(parameters=parameters):
                self.assertEqual(self.validate(job)["parameters"], parameters)
            config = protocol.run_config(protocol.PILOT_TASK, parameters)
            self.assertIs(type(config["seed"]), int)
            self.assertEqual(config["seed"], 0)
            self.assertFalse(config["fitting_allowed"])
            self.assertFalse(config["expansion_authorized"])
            self.assertEqual(config["training_tasks"], 0)
            expected_epoch = 'ose_replay_v1r3_' if parameters['recipe'] == 'fisher_deep' else 'ose_replay_v1_'
            self.assertTrue(config['run_id'].startswith(expected_epoch))
            self.assertEqual(config["run_id"], protocol.run_id(protocol.PILOT_TASK, parameters))
        job = copy.deepcopy(self.job)
        job["parameters"] = dict(recipe="dr_path", dataset="legacy_control_iid")
        with self.assertRaises(fixture.ADAPTER.AdapterError):
            self.validate(job)

    def test_caller_cannot_inject_seeds_paths_resources_or_unknown_values(self):
        cases = []
        for key in ("seed", "seed_start", "seed_end", "shard_index", "shard_count",
                    "resources", "execution_profile", "cpu_cores", "gpu_count",
                    "run_id", "output_dir", "command"):
            parameters = dict(self.job["parameters"])
            parameters[key] = 1
            cases.append(parameters)
        for key in ("recipe", "dataset"):
            parameters = dict(self.job["parameters"])
            del parameters[key]
            cases.append(parameters)
            for value in ("unknown", None, 0, True, [], {}):
                parameters = dict(self.job["parameters"])
                parameters[key] = value
                cases.append(parameters)
        cases.extend((None, [], "fisher_shallow"))
        for parameters in cases:
            job = copy.deepcopy(self.job)
            job["parameters"] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)

    def test_wrong_profile_allocation_version_and_every_missing_environment_field_fail(self):
        for key, value in (("schema_version", 1), ("execution_profile", "gpu"),
                           ("state", "pending"), ("requested_profile", "other")):
            job = copy.deepcopy(self.job)
            job[key] = value
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)
        for key, value in (("cpu_cores", 2), ("memory_mib", 4096), ("gpu_count", 1),
                           ("gpu_memory_mib", 1), ("gpu_utilization_pct", 1)):
            job = copy.deepcopy(self.job)
            job["allocation"][key] = value
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)
        environment = self.helper.environment(self.path, self.job)
        for key in environment:
            missing = dict(environment)
            del missing[key]
            with self.subTest(missing=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(self.job, missing)
        for key, value in (("SERVER_SCHEDULER_ATTEMPT", "0"),
                           ("SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS", "1"),
                           ("SERVER_SCHEDULER_GPU_INDEX", ""),
                           ("SERVER_SCHEDULER_GPU_UUIDS", "physical-device"),
                           ("CUDA_VISIBLE_DEVICES", "0"),
                           ("NVIDIA_VISIBLE_DEVICES", "all")):
            altered = dict(environment, **{key: value})
            with self.subTest(key=key), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(self.job, altered)

    def test_legacy_tasks_and_unproposed_formal_task_cannot_dispatch_this_science(self):
        for task in (fixture.ADAPTER.DSLASSO_TASK, fixture.ADAPTER.ONE_SHOT_TASK,
                     *fixture.ADAPTER.COMPARISON_TASKS, fixture.ADAPTER.ONE_SHOT_EFFECTS_PILOT_TASK,
                     "effect-modifier-effects-replay-shard", "effect-modifier-effects-training-pilot"):
            job = copy.deepcopy(self.job)
            job["task"] = task
            with self.subTest(task=task), self.assertRaises(fixture.ADAPTER.AdapterError):
                self.validate(job)

    def test_schema_and_five_examples_match_the_exact_replay_plan(self):
        schema = json.loads((PROPOSAL_ROOT / "pilot-parameters.schema.json").read_text())
        self.assertEqual(schema["type"], "object")
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(set(schema["required"]), {"recipe", "dataset"})
        self.assertEqual(set(schema["properties"]), {"recipe", "dataset"})
        self.assertEqual(schema["properties"]["recipe"]["enum"],
                         ["fisher_shallow", "fisher_deep", "dr_path"])
        self.assertEqual(schema["properties"]["dataset"]["enum"],
                         ["scmv2_base", "legacy_control_iid"])
        self.assertEqual(schema["oneOf"], [
            {"properties": {"recipe": {"enum": ["fisher_shallow", "fisher_deep"]}}},
            {"properties": {"recipe": {"const": "dr_path"},
                            "dataset": {"const": "scmv2_base"}}},
        ])
        examples = sorted((PROPOSAL_ROOT / "examples").glob("*.job-request.json"))
        self.assertEqual(len(examples), 5)
        observed, job_ids = [], set()
        for path in examples:
            self.assertTrue(path.is_relative_to(fixture.PROJECT_ROOT))
            self.assertNotIn("outbox", path.parts)
            example = json.loads(path.read_text())
            self.assertEqual(set(example), {
                "schema_version", "job_id", "project", "task", "parameters", "priority"
            })
            self.assertEqual(example["schema_version"], 2)
            self.assertEqual(example["project"], "FedFisher")
            self.assertEqual(example["task"], protocol.PILOT_TASK)
            self.assertEqual(example["priority"], 0)
            self.assertRegex(example["job_id"], fixture.ADAPTER.IDENTIFIER)
            self.assertNotIn(example["job_id"], job_ids)
            job_ids.add(example["job_id"])
            self.assertEqual(path.name, example["job_id"] + ".job-request.json")
            observed.append(protocol.validate_parameters(example["task"], example["parameters"]))
        self.assertCountEqual(observed, protocol.cohort_plan())
        plan = json.loads((PROPOSAL_ROOT / "replay-pilot-plan.json").read_text())
        self.assertCountEqual(plan["stage1_replay"], protocol.cohort_plan())
        for field in ("training_tasks_authorized", "compression_authorized", "new_seeds_authorized"):
            self.assertIs(plan[field], False)

    def test_disabled_proposal_has_only_the_pilot_and_truthful_unmeasured_profile(self):
        proposal = (PROPOSAL_ROOT / "registration-proposal.toml").read_text()
        self.assertEqual(re.findall(r"(?m)^enabled = (.+)$", proposal), ["false"])
        self.assertIn("unmeasured admission estimates", proposal)
        self.assertIn("no peak-memory or runtime pilot measurement is claimed", proposal)
        self.assertIn("not outbox requests or submitted jobs", proposal)
        self.assertIn("There is no training, compression, new seed or formal cohort task", proposal)
        self.assertIn('entrypoint = "/home/del6500/projects/FedFisher/scripts/'
                      'fedfisher_server_scheduler_adapter.py"', proposal)
        self.assertIn('allowed_tasks = ["' + protocol.PILOT_TASK + '"]', proposal)
        self.assertEqual(proposal.count("[[tasks]]"), 1)
        self.assertEqual(proposal.count("[[tasks.execution_profiles]]"), 1)
        self.assertIn('name = "' + protocol.PILOT_TASK + '"', proposal)
        for key, value in {
            "allow_gpu_jobs": "false", "kind": '"cpu"', "resource_mode": '"fixed"',
            "cpu_cores_min": "1", "cpu_cores_preferred": "1", "cpu_cores_max": "1",
            "host_memory_mode": '"fixed"', "host_memory_min_mib": "8192",
            "host_memory_default_mib": "8192", "host_memory_max_mib": "8192",
            "gpu_count": "0", "gpu_memory_mib": "0", "gpu_utilization_pct": "0",
            "estimated_runtime_seconds": str(self.job["estimated_runtime_seconds"]),
        }.items():
            with self.subTest(key=key):
                self.assertEqual(re.findall(r"(?m)^" + key + r" = (.+)$", proposal), [value])
        self.assertNotIn("effect-modifier-effects-replay-shard", proposal)


if __name__ == "__main__":
    unittest.main()
