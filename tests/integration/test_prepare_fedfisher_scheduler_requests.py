from tests.paths import PROJECT_ROOT
import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = PROJECT_ROOT / "effect_modifier_experiment/scheduling/publication/original.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_fedfisher_scheduler_requests", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUESTS = load_module()


class PrepareFedFisherSchedulerRequestsTests(unittest.TestCase):
    def test_pilots_are_bounded_and_cover_historical_failure_shards(self):
        requests = REQUESTS.build_requests("pilot")
        self.assertEqual(len(requests), 8)
        one_shot = {
            payload["parameters"]["variant"]: payload["parameters"]["shard_index"]
            for _, payload in requests
            if payload["task"] == "effect-modifier-one-shot-lasso-shard"
        }
        self.assertEqual(
            one_shot,
            {
                "control": 6,
                "site_gaussian_shift": 13,
                "local_permutation": 6,
                "site_gaussian_local_permutation": 13,
            },
        )

    def test_remaining_plus_pilots_has_exact_disjoint_shard_coverage(self):
        combined = REQUESTS.build_requests("pilot") + REQUESTS.build_requests(
            "remaining"
        )
        self.assertEqual(len(combined), 80)
        for task_label, contract in REQUESTS.TASKS.items():
            del task_label
            for variant in REQUESTS.VARIANTS:
                indices = sorted(
                    payload["parameters"]["shard_index"]
                    for _, payload in combined
                    if payload["task"] == contract["task"]
                    and payload["parameters"]["variant"] == variant
                )
                self.assertEqual(indices, list(range(contract["shard_count"])))

    def test_requests_express_only_scientific_shard_intent(self):
        requests = REQUESTS.build_requests("pilot") + REQUESTS.build_requests(
            "remaining"
        )
        job_ids = []
        for name, payload in requests:
            self.assertEqual(name, payload["job_id"] + ".job-request.json")
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["project"], "FedFisher")
            self.assertEqual(payload["priority"], 0)
            self.assertNotIn("resources", payload)
            self.assertNotIn("execution_profile", payload)
            self.assertEqual(
                set(payload["parameters"]),
                {"variant", "shard_index", "shard_count"},
            )
            self.assertLessEqual(len(payload["job_id"]), 128)
            job_ids.append(payload["job_id"])
        self.assertEqual(len(job_ids), len(set(job_ids)))

    def test_remaining_generation_requires_explicit_pilot_gate(self):
        with self.assertRaisesRegex(ValueError, "confirm-pilots-validated"):
            REQUESTS.write_requests("remaining", 1)

    def test_generation_must_be_positive(self):
        for value in (0, -1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    REQUESTS.build_requests("pilot", value)


if __name__ == "__main__":
    unittest.main()
