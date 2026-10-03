"""Real immutable-file/provenance checks with only the scientific hook stubbed."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.dslassocov_run_contract import compute_provenance
from effect_modifier_experiment.one_shot_effects import artifacts
from effect_modifier_experiment.one_shot_effects.wire import canonical, communication_report, pack_payload


SCRATCH = Path("/scr/del6500/FedFisher/one-shot-effects-validation")
HOOK = "effect_modifier_experiment.one_shot_effects.validation"


class EffectArtifactTests(unittest.TestCase):
    def setUp(self):
        artifacts.safe_path(SCRATCH)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="artifact-fixture-", dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
        def git(*arguments):
            return subprocess.run(["git", "-c", "user.name=Artifact Fixture",
                                   "-c", "user.email=artifact-fixture@example.invalid",
                                   "-c", "commit.gpgsign=false", "-c", "core.hooksPath=" + str(self.root / "no-hooks"),
                                   *arguments], cwd=self.repo, env=environment, check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.git = git
        git("init", "-b", "codex/artifact-fixture")
        (self.repo / "fixture.txt").write_text("frozen fixture source\n")
        git("add", "fixture.txt")
        git("commit", "-m", "Create artifact provenance fixture")
        self.provenance = compute_provenance(self.repo)
        self.source_patch = mock.patch.object(artifacts, "PROJECT_ROOT", self.repo)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)
        module = ModuleType(HOOK)
        def semantic(result, arrays, payloads, config, identity):
            if result.get("semantic_valid") is not True:
                raise ValueError("fixture semantic rejection")
        self.validator = mock.Mock(side_effect=semantic)
        module.validate_result = self.validator
        self.module_patch = mock.patch.dict(sys.modules, {HOOK: module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.config = dict(run_id="one_shot_effects_fixture_v1", task="fixture-pilot",
                           recipe="fixture", dataset="SCMV2-base", seed=0,
                           optimizer=dict(tolerance=1e-8))
        self.path = self.root / self.config["run_id"]
        self.store = artifacts.EffectArtifactStore(self.config, self.provenance, self.path)
        self.arrays = {"model_a__coefficients": np.array([1.25, -0.5]),
                       "model_b__weights": np.zeros((2, 3), dtype=np.float32)}
        self.payloads = [pack_payload("fixture", {"summary": np.array([i, 1.0])},
                                      dict(site_id=i, n_total=20+i)) for i in range(2)]
        self.result = dict(identity=self.store.identity, status="complete", semantic_valid=True,
                           communication=communication_report(self.payloads, 17, 29), models={})

    def complete(self):
        self.store.initialize()
        self.store.start(dict(job_id="fixture-job", execution_profile="fixture"))
        self.store.complete(self.result, self.arrays, self.payloads)

    def rehash_result(self, mutate):
        path = self.store.unit_dir / "result.json"
        result = json.loads(path.read_text())
        mutate(result)
        path.write_text(canonical(result))
        marker_path = self.store.unit_dir / "done.json"
        marker = json.loads(marker_path.read_text())
        marker["artifacts"]["result.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        marker_path.write_text(canonical(marker))

    def test_complete_roundtrip_exact_inventory_and_readonly_status(self):
        self.assertEqual(self.store.status(), "missing")
        self.assertFalse(self.path.exists())
        self.complete()
        self.assertEqual(self.store.status(), "validated")
        files = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.path.rglob("*") if p.is_file()}
        self.assertEqual(self.store.status(), "validated")
        self.assertEqual(files, {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files})
        marker = json.loads((self.store.unit_dir / "done.json").read_text())
        self.assertEqual(set(marker["artifacts"]),
                         {"started.json", "model.npz", "result.json", "site_000.payload", "site_001.payload"})
        for filename, digest in marker["artifacts"].items():
            self.assertEqual(hashlib.sha256((self.store.unit_dir / filename).read_bytes()).hexdigest(), digest)
        self.assertGreaterEqual(self.validator.call_count, 4)
        with self.assertRaises((ValueError, FileExistsError)):
            self.store.start({})
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays, self.payloads)
        self.assertEqual(files, {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files})

    def test_source_changes_are_detected_from_real_git_and_hashes(self):
        self.complete()
        original = (self.repo / "fixture.txt").read_bytes()
        (self.repo / "fixture.txt").write_text("changed source\n")
        self.assertNotEqual(compute_provenance(self.repo), self.provenance)
        with self.assertRaises(ValueError):
            self.store.status()
        (self.repo / "fixture.txt").write_bytes(original)
        self.assertEqual(self.store.status(), "validated")
        (self.repo / "new-source.py").write_text("SOURCE = 1\n")
        with self.assertRaises(ValueError):
            self.store.status()
        (self.repo / "new-source.py").unlink()
        self.git("commit", "--allow-empty", "-m", "Different clean provenance")
        with self.assertRaises(ValueError):
            artifacts.EffectArtifactStore(self.config, self.provenance, self.path)

    def test_configuration_manifest_and_directory_identity_fail_closed(self):
        self.complete()
        changed = json.loads(canonical(self.config))
        changed["optimizer"]["tolerance"] = 1e-3
        other = artifacts.EffectArtifactStore(changed, self.provenance, self.path)
        with self.assertRaises(ValueError):
            other.status()
        with self.assertRaises(ValueError):
            artifacts.EffectArtifactStore(self.config, self.provenance, self.root / "other-name")
        manifest = self.path / "run_manifest.env"
        manifest.write_text(manifest.read_text().replace("format_version=2", "format_version=1"))
        with self.assertRaises(ValueError):
            self.store.status()

    def test_every_file_hash_and_exact_inventory_is_enforced(self):
        self.complete()
        for name in ("started.json", "model.npz", "result.json", "site_000.payload"):
            with self.subTest(name=name):
                path = self.store.unit_dir / name
                old = path.read_bytes()
                path.write_bytes(old+b"tamper")
                with self.assertRaises(ValueError):
                    self.store.status()
                path.write_bytes(old)
        extra = self.store.unit_dir / "unexpected.tmp"
        extra.write_text("partial")
        with self.assertRaises(ValueError):
            self.store.status()
        extra.unlink()
        self.assertEqual(self.store.status(), "validated")

    def test_rehashed_semantic_corruption_is_not_completion(self):
        self.complete()
        self.rehash_result(lambda result: result.update(semantic_valid=False))
        with self.assertRaisesRegex(ValueError, "semantic rejection"):
            self.store.status()

    def test_rehashed_wire_accounting_corruption_is_rejected(self):
        self.complete()
        self.rehash_result(lambda result: result["communication"].update(upload_bytes=1))
        with self.assertRaisesRegex(ValueError, "wire payload"):
            self.store.status()

    def test_complete_flag_and_identity_are_required_before_writing(self):
        self.store.initialize()
        self.store.start({})
        for changes in ({"status": "running"}, {"identity": dict(self.store.identity, seed=1)},
                        {"communication": communication_report([b"different"])}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                self.store.complete(dict(self.result, **changes), self.arrays, self.payloads)
            self.assertEqual({p.name for p in self.store.unit_dir.iterdir()}, {"started.json"})
        with self.assertRaises(ValueError):
            self.store.status()

    def test_partial_publication_is_preserved_without_done_or_overwrite(self):
        self.store.initialize()
        self.store.start({})
        real_create = artifacts._atomic_create
        def fail_result(path, data):
            if Path(path).name == "result.json":
                raise OSError("simulated publication interruption")
            return real_create(path, data)
        with mock.patch.object(artifacts, "_atomic_create", side_effect=fail_result), self.assertRaises(OSError):
            self.store.complete(self.result, self.arrays, self.payloads)
        self.assertTrue((self.store.unit_dir / "model.npz").is_file())
        self.assertFalse((self.store.unit_dir / "done.json").exists())
        before = (self.store.unit_dir / "model.npz").read_bytes()
        with self.assertRaises(ValueError):
            self.store.status()
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays, self.payloads)
        self.assertEqual((self.store.unit_dir / "model.npz").read_bytes(), before)

    def test_symlinks_and_nonempty_manifestless_directories_are_rejected(self):
        target = self.root / "real-output"
        target.mkdir()
        self.path.symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            artifacts.EffectArtifactStore(self.config, self.provenance, self.path)
        self.path.unlink()
        self.path.mkdir()
        (self.path / "leftover").write_text("preserve")
        with self.assertRaises(ValueError):
            self.store.initialize()
        self.assertEqual((self.path / "leftover").read_text(), "preserve")

    def test_payload_symlink_rejected_even_when_target_bytes_match(self):
        self.complete()
        payload = self.store.unit_dir / "site_000.payload"
        target = self.root / "same-payload"
        target.write_bytes(payload.read_bytes())
        payload.unlink()
        payload.symlink_to(target)
        with self.assertRaises(ValueError):
            self.store.status()

    def test_failure_evidence_requires_started_identity_and_remains_partial(self):
        self.store.initialize()
        with self.assertRaises(ValueError):
            self.store.fail(ValueError("not started"))
        self.store.start({})
        self.store.fail(RuntimeError("numerical failure"))
        failure = self.store.unit_dir / "failure.json"
        before = failure.read_bytes()
        self.assertEqual(json.loads(before), dict(identity=self.store.identity,
                                                  error_type="RuntimeError", message="numerical failure"))
        with self.assertRaises(FileExistsError):
            self.store.fail(ValueError("cannot replace"))
        with self.assertRaises(ValueError):
            self.store.status()
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays, self.payloads)
        self.assertEqual(failure.read_bytes(), before)
        other_config = dict(self.config, run_id="one_shot_effects_completed_fixture_v1")
        other = artifacts.EffectArtifactStore(other_config, self.provenance,
                                              self.root / other_config["run_id"])
        other.initialize()
        other.start({})
        other.complete(dict(self.result, identity=other.identity), self.arrays, self.payloads)
        with self.assertRaises(FileExistsError):
            other.fail(ValueError("completed already"))
        self.assertFalse((other.unit_dir / "failure.json").exists())
        self.assertEqual(other.status(), "validated")


if __name__ == "__main__":
    unittest.main()
