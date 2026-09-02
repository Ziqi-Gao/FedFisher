import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
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
            "SERVER_SCHEDULER_ATTEMPT": "1",
            "SERVER_SCHEDULER_EXECUTION_PROFILE": payload["execution_profile"],
            "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS": "%.6f"
            % payload["estimated_runtime_seconds"],
            "SERVER_SCHEDULER_CPU_CORES": str(allocation["cpu_cores"]),
            "SERVER_SCHEDULER_CPUSET": ",".join(
                str(value) for value in payload["cpu_ids"]
            ),
            "SERVER_SCHEDULER_MEMORY_MIB": str(allocation["memory_mib"]),
            "SERVER_SCHEDULER_GPU_COUNT": str(allocation["gpu_count"]),
            "SERVER_SCHEDULER_GPU_MEMORY_MIB": str(
                allocation["gpu_memory_mib"]
            ),
            "SERVER_SCHEDULER_GPU_UTILIZATION_PCT": str(
                allocation["gpu_utilization_pct"]
            ),
            "SERVER_SCHEDULER_GPU_EXCLUSIVITY": (
                "exclusive" if allocation["exclusive_gpu"] else "shared"
            ),
            "SERVER_SCHEDULER_GPU_INDICES": ",".join(
                str(value) for value in payload["gpu_indices"]
            ),
            "SERVER_SCHEDULER_GPU_UUIDS": ",".join(payload["gpu_uuids"]),
            "SERVER_SCHEDULER_GPU_PCI_BUS_IDS": ",".join(
                payload["gpu_pci_bus_ids"]
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
                self.assertEqual(validated["schema_version"], 2)
                self.assertEqual(validated["execution_profile"], "cpu-1core")

    def test_normal_v2_request_may_omit_resources(self):
        path, payload = self.fixture("dslassocov-running-job.json")
        self.assertIsNone(payload["resources"])
        validated = ADAPTER.load_running_job(path, self.environment(path, payload))
        self.assertIsNone(validated["resources"])

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
        caller_run_id = copy.deepcopy(payload)
        caller_run_id["parameters"]["run_id"] = "caller-controlled"
        cases.append(caller_run_id)
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
        wrong_profile = copy.deepcopy(payload)
        wrong_profile["execution_profile"] = "cpu-other"
        for candidate in (
            invalid_index,
            invalid_count,
            too_many_cores,
            gpu_request,
            wrong_profile,
        ):
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

    def test_protocol_v2_environment_is_fail_closed(self):
        path, payload = self.fixture("dslassocov-running-job.json")
        cases = []
        bad_attempt = self.environment(path, payload)
        bad_attempt["SERVER_SCHEDULER_ATTEMPT"] = "0"
        cases.append(bad_attempt)
        wrong_profile = self.environment(path, payload)
        wrong_profile["SERVER_SCHEDULER_EXECUTION_PROFILE"] = "cpu-other"
        cases.append(wrong_profile)
        wrong_runtime = self.environment(path, payload)
        wrong_runtime["SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS"] = "1.0"
        cases.append(wrong_runtime)
        host_gpu = self.environment(path, payload)
        host_gpu["SERVER_SCHEDULER_GPU_INDICES"] = "0"
        cases.append(host_gpu)
        singular_v1 = self.environment(path, payload)
        singular_v1["SERVER_SCHEDULER_GPU_INDEX"] = ""
        cases.append(singular_v1)
        for environment in cases:
            with self.subTest(environment=environment):
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
            all(
                "server_scheduler_v2r1_300seeds" in value
                for value in ADAPTER.DSLASSO_RUN_IDS.values()
            )
        )
        self.assertNotEqual(
            ADAPTER.DSLASSO_RUN_IDS["control"],
            "dslassocov_control_server_scheduler_v2_300seeds",
        )
        self.assertEqual(
            ADAPTER.ONE_SHOT_RUN_SUFFIX,
            "stationarity_polish_full_scope_server_scheduler_v2r1_300seeds",
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
        self.assertIn("server_scheduler_v2", one_shot.run_id)
        self.assertEqual(one_shot.threads, 1)

    def test_v2_examples_use_automatic_profile_selection(self):
        example_root = (
            PROJECT_ROOT / "effect_modifier_experiment/server_scheduler/examples"
        )
        for name in (
            "dslassocov-shard.job-request.json",
            "one-shot-lasso-shard.job-request.json",
        ):
            with self.subTest(name=name):
                payload = json.loads((example_root / name).read_text(encoding="utf-8"))
                self.assertEqual(payload["schema_version"], 2)
                self.assertNotIn("resources", payload)
                self.assertNotIn("execution_profile", payload)
                self.assertEqual(
                    set(payload["parameters"]),
                    {"variant", "shard_index", "shard_count"},
                )

    def test_retry_skips_validated_dslasso_seeds_and_completes_only_missing_seed(self):
        from effect_modifier_experiment import manage_dslassocov_cpu_run as manager
        from effect_modifier_experiment import run_dslassocov_effect_modifier as runner

        job = {
            "parameters": {
                "variant": "control",
                "shard_index": 0,
                "shard_count": 4,
            }
        }
        output_dir = Path(
            "/data/del6500/FedFisher/effect_modifier_experiment/outputs/fixture"
        )

        def seed_status(unused_run_id, unused_variant, seed):
            return 3 if seed == 0 else 0

        with mock.patch.object(
            manager, "initialize_run", return_value=({"output_dir": output_dir}, {})
        ), mock.patch.object(manager, "validate_current_run"), mock.patch.object(
            manager, "seed_status", side_effect=seed_status
        ), mock.patch.object(manager, "complete_seed") as complete_seed, mock.patch.object(
            runner, "run"
        ) as run, mock.patch.object(
            ADAPTER,
            "_dslasso_runner_args",
            return_value=SimpleNamespace(seed=0, split="iid"),
        ):
            self.assertEqual(ADAPTER._run_dslasso(job), 0)
        self.assertEqual(run.call_count, 2)
        complete_seed.assert_called_once_with(
            ADAPTER.DSLASSO_RUN_IDS["control"], "control", 0
        )

    def test_retry_skips_validated_one_shot_cells_and_revalidates_new_marker(self):
        from effect_modifier_experiment import manage_one_shot_lasso_cpu_run as manager
        from effect_modifier_experiment import run_one_shot_lasso_effect_modifier as runner

        job = {
            "parameters": {
                "variant": "control",
                "shard_index": 0,
                "shard_count": 16,
            }
        }
        target_calls = 0

        def cell_status(args):
            nonlocal target_calls
            is_target = (
                args.seed == 0
                and args.family == "one_shot_local_logistic_lasso_fedavg"
                and args.split == "iid"
                and args.model == "linear"
            )
            if not is_target:
                return 0
            target_calls += 1
            return manager.MISSING_EXIT if target_calls == 1 else 0

        with mock.patch.object(manager, "initialize_formal"), mock.patch.object(
            manager, "cell_status", side_effect=cell_status
        ), mock.patch.object(runner, "run") as run:
            self.assertEqual(ADAPTER._run_one_shot(job), 0)
        run.assert_called_once()
        self.assertEqual(target_calls, 2)

    def test_adapter_runtime_and_scientific_roots_are_project_owned(self):
        self.assertEqual(
            ADAPTER.PROJECT_ROOT, Path("/home/del6500/projects/FedFisher")
        )
        self.assertEqual(ADAPTER.SCRATCH_ROOT, Path("/scr/del6500/FedFisher"))
        source = ADAPTER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("/data/del6500/ServerScheduler", source)
        self.assertNotIn("/scr/del6500/ServerScheduler", source)


if __name__ == "__main__":
    unittest.main()
