import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "scripts/fedfisher_server_scheduler_adapter.py"
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/server_scheduler"
TEST_SCRATCH_ROOT = Path(
    "/scr/del6500/FedFisher/effect_modifier_experiment/server_scheduler_fixture_tests"
)


def load_adapter():
    spec = importlib.util.spec_from_file_location("fedfisher_scheduler_adapter", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_adapter()


class FedFisherServerSchedulerAdapterTests(unittest.TestCase):
    def fixture(self, name):
        path = (FIXTURE_ROOT / name).resolve(strict=True)
        return path, json.loads(path.read_text(encoding="utf-8"))

    def environment(self, path, payload):
        allocation = payload["allocation"]
        environment = {
            "SERVER_SCHEDULER_JOB_ID": payload["job_id"],
            "SERVER_SCHEDULER_PROJECT": payload["project"],
            "SERVER_SCHEDULER_JOB_MANIFEST": str(path),
            "SERVER_SCHEDULER_LEASE_ID": "lease-fixture",
            "SERVER_SCHEDULER_CPU_CORES": str(allocation["cpu_cores"]),
            "SERVER_SCHEDULER_CPUSET": ",".join(
                str(value) for value in payload["cpu_ids"]
            ),
            "SERVER_SCHEDULER_MEMORY_MIB": str(allocation["memory_mib"]),
            "SERVER_SCHEDULER_GPU_COUNT": str(allocation["gpu_count"]),
            "SERVER_SCHEDULER_GPU_MEMORY_MIB": str(
                allocation["gpu_memory_mib"]
            ),
            "SERVER_SCHEDULER_STDOUT_LOG": payload["stdout_log"],
            "SERVER_SCHEDULER_STDERR_LOG": payload["stderr_log"],
        }
        if payload["numa_node"] is not None:
            environment["SERVER_SCHEDULER_NUMA_NODE"] = str(payload["numa_node"])
        return environment

    def test_both_fixture_manifests_validate(self):
        for name in ("dslassocov-running-job.json", "one-shot-running-job.json"):
            with self.subTest(name=name):
                path, payload = self.fixture(name)
                validated = ADAPTER.load_running_job(
                    path, self.environment(path, payload)
                )
                self.assertEqual(validated["task"], payload["task"])
                self.assertEqual(validated["parameters"], payload["parameters"])

    def test_dispatch_is_fixed_and_executor_can_be_replaced_without_science(self):
        path, payload = self.fixture("one-shot-running-job.json")
        received = []

        def fixture_executor(job):
            received.append(job)
            return 17

        exit_code = ADAPTER.run_from_manifest(
            path, self.environment(path, payload), executor=fixture_executor
        )
        self.assertEqual(exit_code, 17)
        self.assertEqual(received[0]["task"], ADAPTER.ONE_SHOT_TASK)

    def test_wrong_project_task_state_and_command_parameter_fail_closed(self):
        path, payload = self.fixture("dslassocov-running-job.json")
        cases = []
        wrong_project = copy.deepcopy(payload)
        wrong_project["project"] = "OtherProject"
        cases.append(wrong_project)
        wrong_task = copy.deepcopy(payload)
        wrong_task["task"] = "shell"
        cases.append(wrong_task)
        wrong_state = copy.deepcopy(payload)
        wrong_state["state"] = "pending"
        cases.append(wrong_state)
        command = copy.deepcopy(payload)
        command["parameters"]["command"] = "touch forbidden"
        cases.append(command)
        for candidate in cases:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ADAPTER.AdapterError):
                    ADAPTER.validate_running_job(
                        candidate, self.environment(path, candidate), path
                    )

    def test_parameter_and_allocation_envelopes_fail_closed(self):
        path, payload = self.fixture("one-shot-running-job.json")
        invalid_index = copy.deepcopy(payload)
        invalid_index["parameters"]["shard_index"] = 16
        invalid_count = copy.deepcopy(payload)
        invalid_count["parameters"]["shard_count"] = 32
        too_many_cores = copy.deepcopy(payload)
        too_many_cores["allocation"]["cpu_cores"] = 2
        too_many_cores["cpu_ids"] = [64, 65]
        gpu_request = copy.deepcopy(payload)
        gpu_request["resources"]["gpu_count"] = 1
        gpu_request["resources"]["gpu_memory_mib"] = 1024
        gpu_request["resources"]["gpu_utilization_pct"] = 10
        for candidate in (invalid_index, invalid_count, too_many_cores, gpu_request):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ADAPTER.AdapterError):
                    ADAPTER.validate_running_job(
                        candidate, self.environment(path, candidate), path
                    )

    def test_environment_must_match_final_allocation(self):
        path, payload = self.fixture("dslassocov-running-job.json")
        environment = self.environment(path, payload)
        environment["SERVER_SCHEDULER_CPUSET"] = "7"
        with self.assertRaises(ADAPTER.AdapterError):
            ADAPTER.validate_running_job(payload, environment, path)

    def test_symlinked_manifest_is_rejected(self):
        TEST_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
        path, payload = self.fixture("dslassocov-running-job.json")
        with tempfile.TemporaryDirectory(
            prefix="manifest_symlink_", dir=TEST_SCRATCH_ROOT
        ) as temporary:
            link = Path(temporary) / "running.json"
            link.symlink_to(path)
            with self.assertRaises(ADAPTER.AdapterError):
                ADAPTER.load_running_job(link, self.environment(path, payload))

    def test_adapter_has_no_legacy_resource_scheduler_calls(self):
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "screen -",
            "nvidia-smi",
            "flock",
            "server_scheduler submit",
            "effect_modifier_dslassocov_cpu_lock",
            "effect_modifier_one_shot_cpu_lock",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_dslasso_manager_keeps_legacy_default_but_adapter_can_skip_dirs(self):
        manager_source = (
            PROJECT_ROOT / "effect_modifier_experiment/manage_dslassocov_cpu_run.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "def initialize_run(run_id, variant, create_legacy_scheduler_dirs=True)",
            manager_source,
        )
        self.assertIn("create_legacy_scheduler_dirs=False", ADAPTER_PATH.read_text())

    def test_scheduler_run_names_do_not_collide_with_preserved_legacy_results(self):
        self.assertTrue(
            all("server_scheduler_v1" in value for value in ADAPTER.DSLASSO_RUN_IDS.values())
        )
        self.assertIn("server_scheduler_v1", ADAPTER.ONE_SHOT_RUN_SUFFIX)
        self.assertNotEqual(
            ADAPTER.ONE_SHOT_RUN_SUFFIX,
            "stationarity_polish_sharded_v1_300seeds",
        )

    def test_scientific_argument_builders_keep_fixed_formal_contracts(self):
        from effect_modifier_experiment import manage_one_shot_lasso_cpu_run
        from effect_modifier_experiment import run_dslassocov_effect_modifier

        with mock.patch.dict(os.environ, {"CPU_WORKERS": "2"}):
            dslasso = ADAPTER._dslasso_runner_args(
                run_dslassocov_effect_modifier,
                ADAPTER.DSLASSO_RUN_IDS["control"],
                "control",
                Path("/data/del6500/FedFisher/effect_modifier_experiment/outputs")
                / ADAPTER.DSLASSO_RUN_IDS["control"],
            )
        self.assertEqual(dslasso.threads_per_worker, 1)
        self.assertEqual(dslasso.cpu_workers, 2)
        self.assertEqual(dslasso.seed, 0)
        one_shot = manage_one_shot_lasso_cpu_run.formal_runner_args(
            "one_shot_local_logistic_lasso_fedavg",
            "control",
            run_suffix=ADAPTER.ONE_SHOT_RUN_SUFFIX,
        )
        self.assertIn("server_scheduler_v1", one_shot.run_id)
        self.assertEqual(one_shot.threads, 1)


if __name__ == "__main__":
    unittest.main()
