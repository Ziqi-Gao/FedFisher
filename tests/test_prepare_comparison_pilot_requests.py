"""Request-only fixtures; never write the real outbox or submit computation."""

from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_comparison_pilot_requests", ROOT / "scripts/prepare_comparison_pilot_requests.py")
REQUESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REQUESTS)


def provenance(**changes):
    result = dict(git_commit="a" * 40, git_dirty="0",
                  git_status_sha256=hashlib.sha256(b"").hexdigest(),
                  source_tree_sha256="b" * 64, provenance_sha256="c" * 64)
    result.update(changes)
    return result


class ComparisonPilotRequestTests(unittest.TestCase):
    def setUp(self):
        scratch = Path("/scr/del6500/FedFisher/comparison-request-tests")
        REQUESTS._safe_path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temporary = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.outbox = self.temporary / "scheduler/outbox"
        self.reports = self.temporary / "scheduler/reports"
        self.bundle = self.outbox / REQUESTS.bundle_name(1)
        self.stack.enter_context(mock.patch.object(REQUESTS, "OUTBOX_ROOT", self.outbox))
        self.stack.enter_context(mock.patch.object(REQUESTS, "REPORT_ROOT", self.reports))
        self.provenance = self.stack.enter_context(mock.patch.object(
            REQUESTS, "compute_provenance", return_value=provenance()))

    def test_exact_pilot_coverage_and_scientific_only_requests(self):
        specs = REQUESTS.build_requests(1)
        self.assertEqual(len(specs), 96)
        coverage = set()
        ids = set()
        for item in specs:
            request = item["request"]
            self.assertEqual(set(request), {"schema_version", "job_id", "project", "task", "parameters", "priority"})
            self.assertEqual((request["schema_version"], request["project"], request["priority"]), (2, "FedFisher", 0))
            self.assertEqual(request["task"], "effect-modifier-comparison-pilot")
            self.assertEqual(set(request["parameters"]), {"variant", "cell", "shard_index", "shard_count"})
            self.assertEqual(request["parameters"]["shard_index"], 0)
            self.assertEqual(request["parameters"]["shard_count"], 1)
            self.assertEqual(item["seed_sequence"], [0])
            self.assertEqual(item["splits"], ["iid", "noniid"])
            self.assertEqual(item["filename"], request["job_id"] + ".job-request.json")
            self.assertLessEqual(len(request["job_id"]), 128)
            for forbidden in ("gpu", "cpu", "memory", "resource", "exclusive", "device"):
                self.assertNotIn(forbidden, request["job_id"])
            ids.add(request["job_id"])
            coverage.add((request["parameters"]["cell"], request["parameters"]["variant"]))
            self.assertEqual(item["request_sha256"], hashlib.sha256(REQUESTS._encoded(request)).hexdigest())
        self.assertEqual(len(ids), 96)
        self.assertEqual(coverage, {(cell, variant) for cell in REQUESTS.protocol.CELLS for variant in REQUESTS.protocol.VARIANTS})
        self.assertFalse(self.outbox.exists())

    def test_generation_is_strict_and_has_disjoint_identifiers(self):
        for value in (0, -1, True, False, 1.5, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                REQUESTS.build_requests(value)
        self.assertEqual(REQUESTS.inventory_name(3), "fedfisher-comparison-v1-pilot-cohort-p3.inventory.json")
        first = {item["filename"] for item in REQUESTS.build_requests(1)}
        second = {item["filename"] for item in REQUESTS.build_requests(2)}
        self.assertFalse(first & second)

    def test_dirty_source_rejects_before_any_directory_or_file_write(self):
        self.provenance.return_value = provenance(git_dirty="1")
        with self.assertRaisesRegex(ValueError, "dirty"):
            REQUESTS.write_requests(1)
        self.assertFalse(self.outbox.exists())
        self.assertFalse(self.reports.exists())

    def test_inventory_is_published_last_with_byte_and_config_hashes(self):
        created = []
        original = REQUESTS._atomic_create

        def record(path, data):
            created.append(path)
            original(path, data)

        with mock.patch.object(REQUESTS, "_atomic_create", side_effect=record):
            inventory = REQUESTS.write_requests(1)
        inventory_path = self.reports / REQUESTS.inventory_name(1)
        self.assertEqual(created[-1], inventory_path)
        self.assertEqual(len(created), 97)
        self.assertEqual(json.loads(inventory_path.read_text()), inventory)
        self.assertFalse(inventory["submitted"])
        self.assertTrue(inventory["requires_operator_registration"])
        self.assertTrue(inventory["bundle_ready"])
        self.assertEqual(inventory["publication_mode"], "nested_operator_review_bundle_not_automatic_intake")
        self.assertEqual(inventory["provenance"], provenance())
        self.assertEqual(inventory["scientific_cell_count"], 192)
        self.assertEqual(set(inventory["review_sources"]), {"proposal", "pilot_schema", "adapter", "protocol", "request_builder"})
        for record in inventory["requests"]:
            path = Path(record["path"])
            self.assertEqual(path.parent, self.bundle)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertEqual(json.loads(path.read_text())["parameters"], record["parameters"])
            config_hash = hashlib.sha256(REQUESTS._canonical(record["configuration"]).encode()).hexdigest()
            self.assertEqual(config_hash, record["configuration_sha256"])
        self.assertEqual(len(list(self.bundle.iterdir())), 96)
        self.assertEqual(list(self.outbox.glob("*.json")), [])
        self.assertEqual(self.provenance.call_count, 2)
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(1)
        self.assertEqual(json.loads(inventory_path.read_text()), inventory)

    def test_existing_single_target_or_inventory_rejects_entire_bundle(self):
        self.bundle.mkdir(parents=True)
        first = self.bundle / REQUESTS.build_requests(1)[0]["filename"]
        first.write_text("historical request")
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(1)
        self.assertEqual(first.read_text(), "historical request")
        self.assertEqual(len(list(self.bundle.iterdir())), 1)
        self.assertFalse(self.reports.exists())
        self.reports.mkdir(parents=True)
        old_inventory = self.reports / REQUESTS.inventory_name(2)
        old_inventory.write_text("historical inventory")
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(2)
        self.assertEqual(old_inventory.read_text(), "historical inventory")
        self.assertEqual(len(list(self.bundle.iterdir())), 1)

    def test_symlink_parent_is_rejected_before_directory_creation(self):
        real = self.temporary / "real"
        real.mkdir()
        link = self.temporary / "linked"
        link.symlink_to(real, target_is_directory=True)
        with mock.patch.object(REQUESTS, "OUTBOX_ROOT", link / "outbox"):
            with self.assertRaisesRegex(ValueError, "symlink"):
                REQUESTS.write_requests(1)
        self.assertFalse((real / "outbox").exists())
        self.assertFalse(self.reports.exists())

    def test_changed_source_preserves_requests_without_ready_inventory(self):
        self.provenance.side_effect = [provenance(), provenance(source_tree_sha256="d" * 64)]
        with self.assertRaisesRegex(ValueError, "source changed"):
            REQUESTS.write_requests(1)
        self.assertEqual(len(list(self.bundle.iterdir())), 96)
        self.assertFalse((self.reports / REQUESTS.inventory_name(1)).exists())
        self.provenance.side_effect = None
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(1)

    def test_unexpected_failure_preserves_partial_bundle_and_rejects_retry(self):
        original = REQUESTS._atomic_create
        calls = 0

        def fail_after_one(path, data):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected write interruption")
            original(path, data)

        with mock.patch.object(REQUESTS, "_atomic_create", side_effect=fail_after_one):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                REQUESTS.write_requests(1)
        self.assertEqual(len(list(self.bundle.iterdir())), 1)
        self.assertFalse((self.reports / REQUESTS.inventory_name(1)).exists())
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(1)

    def test_atomic_publish_failure_preserves_pending_file(self):
        self.outbox.mkdir(parents=True)
        target = self.outbox / "fixture.job-request.json"
        with mock.patch.object(REQUESTS.os, "link", side_effect=OSError("injected publish failure")):
            with self.assertRaises(OSError):
                REQUESTS._atomic_create(target, b"fixture")
        self.assertFalse(target.exists())
        self.assertEqual(REQUESTS._pending_path(target).read_bytes(), b"fixture")
        with self.assertRaises(FileExistsError):
            REQUESTS._atomic_create(target, b"replacement")


if __name__ == "__main__":
    unittest.main()
