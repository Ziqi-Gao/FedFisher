"""Request-only fixtures; never write the real outbox or submit computation."""
from tests.paths import PROJECT_ROOT

from tests.integration.request_fixtures import bind_review_sources

from contextlib import ExitStack
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock


ROOT = PROJECT_ROOT
SPEC = importlib.util.spec_from_file_location(
    "prepare_fedfisher_timing_requests", ROOT / "effect_modifier_experiment/scheduling/publication/fedfisher_timing.py")
REQUESTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REQUESTS)


def provenance(**changes):
    result = dict(git_commit="a" * 40, git_dirty="0",
                  git_status_sha256=hashlib.sha256(b"").hexdigest(),
                  source_tree_sha256="b" * 64, provenance_sha256="c" * 64)
    result.update(changes)
    return result


class FedFisherTimingRequestTests(unittest.TestCase):
    def setUp(self):
        scratch = Path("/scr/del6500/FedFisher/fedfisher-timing-request-tests")
        REQUESTS._safe_path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        bind_review_sources(self.stack, REQUESTS)
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
        self.assertEqual(len(specs), 16)
        coverage = set()
        ids = set()
        for item in specs:
            request = item["request"]
            self.assertEqual(set(request), {"schema_version", "job_id", "project", "task", "parameters", "priority"})
            self.assertEqual((request["schema_version"], request["project"], request["priority"]), (2, "FedFisher", 0))
            self.assertEqual(request["task"], "effect-modifier-fedfisher-timing-pilot")
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
        self.assertEqual(len(ids), 16)
        self.assertEqual(coverage, {(cell, variant) for cell in REQUESTS.protocol.CELLS for variant in REQUESTS.protocol.VARIANTS})
        self.assertFalse(self.outbox.exists())

    def test_generation_is_strict_and_has_disjoint_identifiers(self):
        for value in (0, -1, True, False, 1.5, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                REQUESTS.build_requests(value)
        self.assertEqual(REQUESTS.inventory_name(3), "fedfisher-timing-v1-pilot-cohort-p3.inventory.json")
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
        self.assertEqual(len(created), 17)
        self.assertEqual(json.loads(inventory_path.read_text()), inventory)
        self.assertFalse(inventory["submitted"])
        self.assertTrue(inventory["requires_operator_registration"])
        self.assertTrue(inventory["bundle_ready"])
        self.assertEqual(inventory["publication_mode"], "nested_operator_review_bundle_not_automatic_intake")
        self.assertEqual(inventory["provenance"], provenance())
        self.assertEqual(inventory["scientific_cell_count"], 32)
        self.assertFalse(inventory["formal_plan"]["publication_authorized"])
        self.assertEqual(inventory["formal_plan"]["scientific_cell_count"], 9600)
        self.assertEqual(inventory["formal_plan"]["allowed_shard_counts"], [8, 16])
        self.assertEqual(set(inventory["review_sources"]), set(REQUESTS.REVIEW_SOURCES))
        for record in inventory["requests"]:
            path = Path(record["path"])
            self.assertEqual(path.parent, self.bundle)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
            self.assertEqual(json.loads(path.read_text())["parameters"], record["parameters"])
            config_hash = hashlib.sha256(REQUESTS._canonical(record["configuration"]).encode()).hexdigest()
            self.assertEqual(config_hash, record["configuration_sha256"])
        self.assertEqual(len(list(self.bundle.iterdir())), 16)
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
        self.assertEqual(len(list(self.bundle.iterdir())), 16)
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

    def test_changed_methods_variants_or_seed_scope_reject_expansion(self):
        for field, value in (("CELLS", list(REQUESTS.protocol.CELLS)[:-1]),
                             ("CELLS", list(REQUESTS.protocol.CELLS)+["fedavg"]),
                             ("VARIANTS", ["control"]), ("SPLITS", ["iid"])):
            with self.subTest(field=field), mock.patch.object(REQUESTS.protocol, field, value):
                with self.assertRaises(ValueError):
                    REQUESTS.build_requests()
        with mock.patch.object(REQUESTS.protocol, "seed_sequence", return_value=range(300)):
            with self.assertRaisesRegex(ValueError, "seed zero"):
                REQUESTS.build_requests()

    def test_arbitrary_task_and_boolean_pilot_confirmation_cannot_bypass_evidence(self):
        for arguments in (("--stage", "formal"), ("--task", REQUESTS.protocol.FORMAL_TASK),
                          ("--confirm-pilots-validated",)):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    REQUESTS.main(arguments)
                self.assertEqual(stopped.exception.code, 2)
        self.assertFalse(self.outbox.exists())
        self.provenance.assert_not_called()

    def test_formal_coverage_is_exact_for_both_reviewed_shard_counts(self):
        for count in (8, 16):
            specs = REQUESTS.build_requests(1, "formal", count)
            self.assertEqual(len(specs), 16*count)
            observed = {}
            for item in specs:
                request = item["request"]
                self.assertEqual(request["task"], REQUESTS.protocol.FORMAL_TASK)
                self.assertEqual(set(request), {"schema_version", "job_id", "project", "task", "parameters", "priority"})
                key = request["parameters"]["cell"], request["parameters"]["variant"]
                observed.setdefault(key, []).extend(item["seed_sequence"])
            self.assertEqual(len(observed), 16)
            self.assertTrue(all(sorted(seeds) == list(range(300)) for seeds in observed.values()))
            self.assertEqual(sum(len(item["seed_sequence"])*2 for item in specs), 9600)
        self.assertEqual(len(REQUESTS.build_requests(1, "formal")), 256)
        for count in (0, 1, 4, True, "8", 32):
            with self.subTest(count=count), self.assertRaises(ValueError):
                REQUESTS.build_requests(1, "formal", count)
        with self.assertRaisesRegex(ValueError, "pilot shard"):
            REQUESTS.build_requests(1, "pilot", 8)
        self.assertFalse(self.outbox.exists())

    def acceptance_fixture(self):
        path = self.temporary / "pilot-acceptance.json"
        accepted = dict(provenance=provenance(), accepted_cells=32)
        path.write_text(json.dumps(accepted))
        module = ModuleType("effect_modifier_experiment.fedfisher_timing.acceptance")
        module.verify_pilot_acceptance = mock.Mock(return_value=accepted)
        self.stack.enter_context(mock.patch.dict(sys.modules, {module.__name__: module}))
        return path, accepted, module.verify_pilot_acceptance

    def test_formal_requires_verified_complete_same_source_pilot_before_writing(self):
        with self.assertRaisesRegex(ValueError, "requires complete"):
            REQUESTS.write_requests(1, "formal")
        path, accepted, verify = self.acceptance_fixture()
        with self.assertRaisesRegex(ValueError, "only for formal"):
            REQUESTS.write_requests(1, "pilot", path)
        verify.side_effect = ValueError("incomplete pilot evidence")
        with self.assertRaisesRegex(ValueError, "incomplete pilot"):
            REQUESTS.write_requests(1, "formal", path)
        verify.side_effect = None
        verify.return_value = dict(accepted, provenance=provenance(git_commit="e"*40))
        with self.assertRaisesRegex(ValueError, "source provenance"):
            REQUESTS.write_requests(1, "formal", path)
        self.assertFalse(self.outbox.exists())

    def test_formal_inventory_binds_verified_acceptance_and_rechecks_it(self):
        path, accepted, verify = self.acceptance_fixture()
        inventory = REQUESTS.write_requests(1, "formal", path, 8)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(inventory["request_count"], 128)
        self.assertEqual(inventory["scientific_cell_count"], 9600)
        self.assertTrue(inventory["formal_plan"]["publication_authorized"])
        self.assertFalse(inventory["submitted"])
        self.assertEqual(inventory["pilot_acceptance"]["evidence"], accepted)
        self.assertEqual(inventory["pilot_acceptance"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(inventory["coverage"]["seeds"], list(range(300)))
        self.assertEqual(Path(inventory["inventory_path"]).name, REQUESTS.inventory_name(1, "formal"))

    def test_changed_pilot_acceptance_retains_partial_formal_bundle_without_ready_inventory(self):
        path, accepted, verify = self.acceptance_fixture()
        verify.side_effect = [accepted, dict(accepted, unexpected_change=True)]
        with self.assertRaisesRegex(ValueError, "acceptance changed"):
            REQUESTS.write_requests(1, "formal", path, 8)
        self.assertEqual(len(list((self.outbox / REQUESTS.bundle_name(1, "formal")).iterdir())), 128)
        self.assertFalse((self.reports / REQUESTS.inventory_name(1, "formal")).exists())


if __name__ == "__main__":
    unittest.main()
