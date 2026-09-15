"""No-fit publication fixtures; never use the real outbox or scheduler."""

from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.one_shot_effects.diagnostics import recovery as RECOVERY
from effect_modifier_experiment.one_shot_effects.diagnostics.recovery import verify_recovery_evidence


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
            epoch = "v1r3" if request["parameters"]["recipe"] == "fisher_deep" else "v1"
            self.assertTrue(item["run_id"].startswith("ose_replay_" + epoch + "_"))
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

    def recovery_fixture(self):
        """Build a tiny frozen audit, never access real scientific output."""
        directory = self.temporary / "frozen-p2"
        directory.mkdir()
        central_root, output_root = directory / "central-jobs", directory / "outputs"
        central_root.mkdir()
        output_root.mkdir()
        self.stack.enter_context(mock.patch.object(RECOVERY, "CENTRAL_JOBS_ROOT", central_root))
        self.stack.enter_context(mock.patch.object(RECOVERY, "OUTPUT_ROOT", output_root))

        def save(path, value):
            body = REQUESTS._encoded(value)
            path.write_bytes(body)
            return dict(path=str(path), sha256=hashlib.sha256(body).hexdigest())

        records, retained, replaced, audited = [], [], [], []
        for spec in REQUESTS.build_requests(2):
            request = spec["request"]
            parameters = request["parameters"]
            run_id = "ose_replay_v1_"+parameters["recipe"]+"_"+parameters["dataset"]+"_pilot"
            binding = save(directory / spec["filename"], request)
            old = dict(binding, job_id=request["job_id"], task=request["task"],
                       parameters=parameters, run_id=run_id)
            records.append(old)
            is_deep = parameters["recipe"] == "fisher_deep"
            job = dict(job_id=request["job_id"], project="FedFisher", task=request["task"],
                       parameters=parameters, state="failed" if is_deep else "completed",
                       exit_code=1 if is_deep else 0)
            central = save(central_root / (request["job_id"]+".json"), job)
            relative_paths = ["run_manifest.env", "seed_000/started.json"]
            if not is_deep or parameters["dataset"] == "legacy_control_iid":
                relative_paths.extend(["seed_000/result.json", "seed_000/done.json"])
            (output_root / run_id / "seed_000").mkdir(parents=True)
            artifacts = [save(output_root / run_id / relative, {"fixture": relative})
                         for relative in relative_paths]
            item = dict(job_id=request["job_id"], parameters=parameters, run_id=run_id,
                        scientific_status="unaccepted_failed_execution" if is_deep else "validated_frozen_source",
                        central_state=job["state"], central_job=central, artifacts=artifacts)
            (replaced if is_deep else retained).append(item)
            audited.append(dict(job_id=request["job_id"], parameters=parameters, run_id=run_id,
                                historical_scientific_acceptance=not is_deep,
                                manifest_and_started_valid=True,
                                artifact_integrity_valid=len(artifacts) == 4,
                                pure_semantic_validation="passed" if len(artifacts) == 4 else "unavailable_no_result",
                                scientific_files={relative: artifact["sha256"]
                                                  for relative, artifact in zip(relative_paths, artifacts)},
                                central=dict(central, state=job["state"], exit_code=job["exit_code"])))
        inventory = dict(generation=2, bundle="fedfisher-effects-replay-v1-pilot",
                         task=REQUESTS.protocol.PILOT_TASK, request_count=5,
                         bundle_ready=True, worktree_status="clean", provenance=provenance(),
                         requests=records)
        inventory_binding = save(directory / "inventory.json", inventory)
        audit = dict(source=dict(bound_provenance=provenance(),
                                 inventory_sha256=inventory_binding["sha256"],
                                 all_tracked_bytes_equal_bound_git_tree=True,
                                 tracked_source_sha256=provenance()["source_tree_sha256"]),
                     all_scientific_artifacts_unchanged_during_audit=True, cells=audited)
        envelope = dict(schema_version=1, cohort="deep-recovery",
                        source_inventory=inventory_binding,
                        audit_evidence=save(directory / "audit.json", audit),
                        retained=retained, replaced=replaced)
        path = directory / "recovery-evidence.json"
        save(path, envelope)
        return path, envelope, save

    def test_deep_recovery_covers_only_two_failed_cells_with_new_job_and_run_ids(self):
        specs = REQUESTS.build_requests(3, "deep-recovery")
        self.assertEqual(len(specs), 2)
        self.assertEqual({tuple(item["request"]["parameters"][key] for key in ("recipe", "dataset"))
                          for item in specs}, REQUESTS.REPLACED_CELLS)
        self.assertTrue(all(item["run_id"].startswith("ose_replay_v1r3_fisher_deep_") for item in specs))
        self.assertTrue(all("-deep-recovery-cohort-p3-" in item["request"]["job_id"] for item in specs))
        self.assertEqual(REQUESTS.inventory_name(3, "deep-recovery"),
                         "fedfisher-effects-replay-v1-deep-recovery-cohort-p3.inventory.json")
        self.assertFalse({item["filename"] for item in specs} &
                         {item["filename"] for item in REQUESTS.build_requests(2)})
        for generation in (1, 2):
            with self.assertRaisesRegex(ValueError, "generation 3"):
                REQUESTS.build_requests(generation, "deep-recovery")
        for cohort in (None, "remaining", "fisher_deep"):
            with self.assertRaisesRegex(ValueError, "cohort"):
                REQUESTS.build_requests(3, cohort)

    def test_recovery_requires_frozen_evidence_and_a_new_namespace_before_writing(self):
        with self.assertRaisesRegex(ValueError, "requires frozen"):
            REQUESTS.write_requests(3, "deep-recovery")
        path, _, _ = self.recovery_fixture()
        with self.assertRaisesRegex(ValueError, "only valid"):
            REQUESTS.write_requests(3, "full", path)
        with mock.patch.object(REQUESTS.protocol, "run_id", side_effect=lambda task, p:
                               "ose_replay_v1_"+p["recipe"]+"_"+p["dataset"]+"_pilot"):
            with self.assertRaisesRegex(ValueError, "new result namespaces"):
                REQUESTS.write_requests(3, "deep-recovery", path)
        self.assertFalse(self.outbox.exists())

    def test_recovery_inventory_binds_retained_and_replaced_files_and_writes_two_only(self):
        path, envelope, _ = self.recovery_fixture()
        self.assertEqual(verify_recovery_evidence(path), envelope)
        before = {str(p): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
        inventory = REQUESTS.write_requests(3, "deep-recovery", path)
        self.assertEqual(inventory["request_count"], 2)
        self.assertEqual(inventory["retained_request_count"], 3)
        self.assertEqual(inventory["coverage"], dict(fisher_replays=2, dr_path_replays=0, seeds=[0]))
        self.assertEqual(inventory["recovery"]["evidence"], envelope)
        self.assertEqual(inventory["recovery"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(inventory["submitted"])
        self.assertFalse(inventory["requires_operator_registration"])
        self.assertTrue(inventory["requires_operator_source_review"])
        self.assertEqual(len(list(self.outbox.rglob("*.job-request.json"))), 2)
        self.assertEqual({str(p): p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}, before)
        self.assertEqual(json.loads(Path(inventory["inventory_path"]).read_text()), inventory)

    def test_recovery_rejects_wrong_partition_status_and_central_termination(self):
        path, envelope, save = self.recovery_fixture()
        for case in ("duplicate", "status", "central-state", "central-identity"):
            changed = json.loads(json.dumps(envelope))
            if case == "duplicate":
                changed["retained"][0] = changed["retained"][1]
            elif case == "status":
                changed["replaced"][0]["scientific_status"] = "validated_frozen_source"
            else:
                record = changed["retained"][0]
                central = json.loads(Path(record["central_job"]["path"]).read_text())
                if case == "central-state":
                    central.update(state="failed", exit_code=1)
                else:
                    central["parameters"] = dict(recipe="fisher_deep", dataset="scmv2_base")
                record["central_job"] = save(path.parent / (case+".json"), central)
            save(path, changed)
            with self.subTest(case=case), self.assertRaises(ValueError):
                REQUESTS.write_requests(3, "deep-recovery", path)
        self.assertFalse(self.outbox.exists())

    def test_recovery_rejects_changed_artifact_or_bound_symlink(self):
        path, envelope, save = self.recovery_fixture()
        artifact = Path(envelope["retained"][0]["artifacts"][0]["path"])
        body = artifact.read_bytes()
        artifact.write_bytes(body+b" ")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_recovery_evidence(path)
        artifact.write_bytes(body)
        target = path.parent / "relocated-artifact.json"
        target.write_bytes(body)
        artifact.unlink()
        artifact.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "ordinary bound"):
            verify_recovery_evidence(path)

    def test_recovery_rejects_rehashed_empty_or_unaccepted_audit(self):
        path, envelope, save = self.recovery_fixture()
        audit_path = Path(envelope["audit_evidence"]["path"])
        original = json.loads(audit_path.read_text())
        for case in ("empty", "source", "inventory", "tracked", "mutated-artifacts",
                     "missing-cell", "unaccepted", "invalid-manifest", "invalid-integrity", "invalid-semantics"):
            changed = json.loads(json.dumps(envelope))
            audit = json.loads(json.dumps(original))
            if case == "empty":
                audit = {"fixture": True}
            elif case == "source":
                audit["source"]["bound_provenance"]["git_commit"] = "e" * 40
            elif case == "inventory":
                audit["source"]["inventory_sha256"] = "e" * 64
            elif case == "tracked":
                audit["source"]["all_tracked_bytes_equal_bound_git_tree"] = False
            elif case == "mutated-artifacts":
                audit["all_scientific_artifacts_unchanged_during_audit"] = False
            elif case == "missing-cell":
                audit["cells"].pop()
            else:
                key, value = {"unaccepted": ("historical_scientific_acceptance", False),
                              "invalid-manifest": ("manifest_and_started_valid", False),
                              "invalid-integrity": ("artifact_integrity_valid", False),
                              "invalid-semantics": ("pure_semantic_validation", "failed")}[case]
                audit["cells"][0][key] = value
            changed["audit_evidence"] = save(audit_path, audit)
            save(path, changed)
            with self.subTest(case=case), self.assertRaises(ValueError):
                REQUESTS.write_requests(3, "deep-recovery", path)
        self.assertFalse(self.outbox.exists())

    def test_recovery_rejects_incomplete_or_relocated_artifact_inventory(self):
        path, envelope, save = self.recovery_fixture()
        audit_path = Path(envelope["audit_evidence"]["path"])
        original = json.loads(audit_path.read_text())
        for case in ("missing-bound-file", "one-file-audit", "relocated-artifact"):
            changed = json.loads(json.dumps(envelope))
            audit = json.loads(json.dumps(original))
            if case == "missing-bound-file":
                changed["retained"][0]["artifacts"].pop()
            elif case == "one-file-audit":
                changed["retained"][0]["artifacts"] = changed["retained"][0]["artifacts"][:1]
                audit["cells"][0]["scientific_files"] = {"run_manifest.env":
                    audit["cells"][0]["scientific_files"]["run_manifest.env"]}
            else:
                binding = changed["retained"][0]["artifacts"][0]
                copy = path.parent / "same-bytes-wrong-location.json"
                copy.write_bytes(Path(binding["path"]).read_bytes())
                binding["path"] = str(copy)
            changed["audit_evidence"] = save(audit_path, audit)
            save(path, changed)
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "scientific file inventory"):
                verify_recovery_evidence(path)

    def test_recovery_rejects_forged_central_location_even_when_audit_and_hash_agree(self):
        path, envelope, save = self.recovery_fixture()
        record = envelope["retained"][0]
        body = json.loads(Path(record["central_job"]["path"]).read_text())
        record["central_job"] = save(path.parent / "forged-central.json", body)
        audit_path = Path(envelope["audit_evidence"]["path"])
        audit = json.loads(audit_path.read_text())
        audit["cells"][0]["central"].update(record["central_job"])
        envelope["audit_evidence"] = save(audit_path, audit)
        save(path, envelope)
        with self.assertRaisesRegex(ValueError, "authoritative job path"):
            REQUESTS.write_requests(3, "deep-recovery", path)
        self.assertFalse(self.outbox.exists())

    def test_recovery_change_during_publication_keeps_two_requests_without_ready_inventory(self):
        path, envelope, _ = self.recovery_fixture()
        original = REQUESTS._atomic_create

        def change_audit(target, data):
            original(target, data)
            if target.name.endswith(".job-request.json"):
                Path(envelope["audit_evidence"]["path"]).write_text('{"changed": true}')

        with mock.patch.object(REQUESTS, "_atomic_create", side_effect=change_audit):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                REQUESTS.write_requests(3, "deep-recovery", path)
        self.assertEqual(len(list(self.outbox.rglob("*.job-request.json"))), 2)
        self.assertFalse((self.reports / REQUESTS.inventory_name(3, "deep-recovery")).exists())


if __name__ == "__main__":
    unittest.main()
