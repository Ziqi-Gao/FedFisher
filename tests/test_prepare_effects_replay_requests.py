"""No-fit publication fixtures; never use the real outbox or scheduler."""

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
    "prepare_effects_replay_requests", ROOT / "scripts/prepare_effects_replay_requests.py")
REQUESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REQUESTS)


def provenance(**changes):
    result = dict(git_commit="a" * 40, git_dirty="0",
                  git_status_sha256=hashlib.sha256(b"").hexdigest(),
                  source_tree_sha256="b" * 64, provenance_sha256="c" * 64)
    result.update(changes)
    return result


class ReplayRequestTests(unittest.TestCase):
    def setUp(self):
        scratch = Path("/scr/del6500/FedFisher/effects-replay-validation/request-tests")
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
        self.inputs = self.stack.enter_context(mock.patch.object(
            REQUESTS, "_verify_inputs", return_value={"fixture_original_artifact_sha256": "d" * 64}))

    def test_exact_five_requests_express_science_and_bind_distinct_original_sources(self):
        specs = REQUESTS.build_requests()
        self.assertEqual(len(specs), 5)
        actual, identifiers = set(), set()
        for item in specs:
            request = item["request"]
            self.assertEqual(set(request), {"schema_version", "job_id", "project", "task", "parameters", "priority"})
            self.assertEqual((request["schema_version"], request["project"], request["priority"]), (2, "FedFisher", 0))
            self.assertEqual(request["task"], "effect-modifier-effects-replay-pilot")
            self.assertEqual(set(request["parameters"]), {"recipe", "dataset"})
            actual.add(tuple(request["parameters"][key] for key in ("recipe", "dataset")))
            identifiers.add(request["job_id"])
            for forbidden in ("cpu", "gpu", "memory", "resource", "device", "exclusive"):
                self.assertNotIn(forbidden, request["job_id"])
            config = item["configuration"]
            self.assertEqual(config["seed"], 0)
            self.assertEqual(config["training_tasks"], 0)
            self.assertIs(config["fitting_allowed"], False)
            self.assertIs(config["expansion_authorized"], False)
            self.assertIn("accepted_report", config)
            self.assertIn("original_source_provenance", config)
            self.assertEqual(len(config["sources"]), 2)
            self.assertEqual(item["run_id"], config["run_id"])
            self.assertTrue(item["run_id"].startswith("ose_replay_v1_"))
            self.assertEqual(item["filename"], request["job_id"] + ".job-request.json")
            self.assertEqual(item["request_sha256"], hashlib.sha256(REQUESTS._encoded(request)).hexdigest())
        self.assertEqual(actual, REQUESTS.EXPECTED_CELLS)
        self.assertEqual(len(identifiers), 5)
        self.assertFalse(self.outbox.exists())
        self.inputs.assert_not_called()

    def test_generation_strictness_and_disjoint_job_identifiers(self):
        for value in (0, -1, True, False, "1", 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                REQUESTS.build_requests(value)
        first = {item["filename"] for item in REQUESTS.build_requests(1)}
        second = {item["filename"] for item in REQUESTS.build_requests(2)}
        self.assertFalse(first & second)
        self.assertEqual(REQUESTS.inventory_name(3), "fedfisher-effects-replay-v1-pilot-cohort-p3.inventory.json")

    def test_changed_protocol_cannot_expand_or_duplicate_the_five_cells(self):
        plan = REQUESTS.protocol.cohort_plan()
        for altered in (plan[:-1], plan + [plan[0]]):
            with mock.patch.object(REQUESTS.protocol, "cohort_plan", return_value=altered):
                with self.assertRaisesRegex(ValueError, "exactly"):
                    REQUESTS.build_requests()

    def test_dirty_source_or_invalid_original_artifact_stops_before_writing(self):
        self.provenance.return_value = provenance(git_dirty="1")
        with self.assertRaisesRegex(ValueError, "dirty"):
            REQUESTS.write_requests()
        self.inputs.assert_not_called()
        self.provenance.return_value = provenance()
        self.inputs.side_effect = ValueError("original hash mismatch")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            REQUESTS.write_requests()
        self.assertFalse(self.outbox.exists())
        self.assertFalse(self.reports.exists())

    def test_inventory_last_and_request_config_original_and_new_source_hashes_match(self):
        created = []
        original = REQUESTS._atomic_create

        def record(path, data):
            created.append(path)
            original(path, data)

        with mock.patch.object(REQUESTS, "_atomic_create", side_effect=record):
            inventory = REQUESTS.write_requests()
        inventory_path = self.reports / REQUESTS.inventory_name()
        self.assertEqual(created[-1], inventory_path)
        self.assertEqual(len(created), 6)
        self.assertEqual(json.loads(inventory_path.read_text()), inventory)
        self.assertEqual(inventory["request_count"], 5)
        self.assertEqual(inventory["coverage"], dict(fisher_replays=4, dr_path_replays=1, seeds=[0]))
        self.assertFalse(inventory["submitted"])
        self.assertFalse(inventory["expansion_authorized"])
        self.assertTrue(inventory["requires_operator_registration"])
        self.assertTrue(inventory["bundle_ready"])
        self.assertEqual(inventory["provenance"], provenance())
        self.assertEqual(inventory["original_inputs"], self.inputs.return_value)
        self.assertEqual(set(inventory["review_sources"]), set(REQUESTS.REVIEW_SOURCES))
        for item in inventory["requests"]:
            path = Path(item["path"])
            self.assertEqual(path.parent, self.bundle)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
            self.assertEqual(json.loads(path.read_text())["parameters"], item["parameters"])
            self.assertEqual(item["configuration_sha256"],
                             hashlib.sha256(REQUESTS._canonical(item["configuration"]).encode()).hexdigest())
        self.assertEqual(len(list(self.bundle.iterdir())), 5)
        self.assertEqual(list(self.outbox.glob("*.json")), [])
        self.assertEqual(self.inputs.call_count, 2)
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests()
        self.assertEqual(json.loads(inventory_path.read_text()), inventory)

    def test_source_or_input_change_preserves_partial_bundle_without_ready_inventory(self):
        for generation, guard in ((1, "source"), (2, "input")):
            with self.subTest(guard=guard):
                self.provenance.side_effect = ([provenance(), provenance(source_tree_sha256="e" * 64)]
                                               if guard == "source" else None)
                self.inputs.side_effect = ([{"hash": "a"}, {"hash": "b"}] if guard == "input" else None)
                with self.assertRaisesRegex(ValueError, "changed"):
                    REQUESTS.write_requests(generation)
                bundle = self.outbox / REQUESTS.bundle_name(generation)
                self.assertEqual(len(list(bundle.iterdir())), 5)
                self.assertFalse((self.reports / REQUESTS.inventory_name(generation)).exists())
        self.provenance.side_effect = None
        self.inputs.side_effect = None
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(1)

    def test_existing_bundle_or_inventory_rejects_without_overwrite(self):
        self.bundle.mkdir(parents=True)
        first = self.bundle / REQUESTS.build_requests()[0]["filename"]
        first.write_text("historical request")
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests()
        self.assertEqual(first.read_text(), "historical request")
        self.reports.mkdir(parents=True)
        old = self.reports / REQUESTS.inventory_name(2)
        old.write_text("historical inventory")
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests(2)
        self.assertEqual(old.read_text(), "historical inventory")
        self.assertFalse((self.outbox / REQUESTS.bundle_name(2)).exists())

    def test_symlink_parent_and_dangling_pending_symlink_are_rejected(self):
        real = self.temporary / "real"
        real.mkdir()
        link = self.temporary / "linked"
        link.symlink_to(real, target_is_directory=True)
        with mock.patch.object(REQUESTS, "OUTBOX_ROOT", link / "outbox"):
            with self.assertRaisesRegex(ValueError, "symlink"):
                REQUESTS.write_requests()
        self.assertFalse((real / "outbox").exists())
        self.outbox.mkdir(parents=True)
        target = self.outbox / "fixture.job-request.json"
        REQUESTS._pending_path(target).symlink_to(self.temporary / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            REQUESTS._atomic_create(target, b"replacement")
        self.assertFalse((self.temporary / "missing").exists())

    def test_interrupted_publication_keeps_partial_files_and_blocks_retry(self):
        original = REQUESTS._atomic_create
        calls = []

        def interrupt(path, data):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("injected interruption")
            original(path, data)

        with mock.patch.object(REQUESTS, "_atomic_create", side_effect=interrupt):
            with self.assertRaisesRegex(OSError, "interruption"):
                REQUESTS.write_requests()
        self.assertEqual(len(list(self.bundle.iterdir())), 1)
        self.assertFalse((self.reports / REQUESTS.inventory_name()).exists())
        with self.assertRaises(FileExistsError):
            REQUESTS.write_requests()

    def test_atomic_failure_preserves_pending_evidence(self):
        self.outbox.mkdir(parents=True)
        target = self.outbox / "fixture.job-request.json"
        with mock.patch.object(REQUESTS.os, "link", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                REQUESTS._atomic_create(target, b"fixture")
        self.assertFalse(target.exists())
        self.assertEqual(REQUESTS._pending_path(target).read_bytes(), b"fixture")
        with self.assertRaises(FileExistsError):
            REQUESTS._atomic_create(target, b"replacement")


if __name__ == "__main__":
    unittest.main()
