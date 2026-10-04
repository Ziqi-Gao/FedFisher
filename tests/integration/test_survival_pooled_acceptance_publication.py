"""Report-proof publication only: no scientific, central, or outbox writes."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure import artifacts as infrastructure
from effect_modifier_experiment.survival_pooled import acceptance as gate, protocol
from scripts import validate_survival_pooled as cli

SCRATCH = Path('/scr/del6500/FedFisher/survival-pooled-validation/acceptance-publication')
REPORT_ROOT = Path('/data/del6500/FedFisher/effect_modifier_experiment/reports/survival-pooled-v1')


def provenance():
    return dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                source_tree_sha256='c' * 64, provenance_sha256='d' * 64)


def proof(cohort='pilot'):
    return dict(schema_version=1, protocol=protocol.PROTOCOL,
        scope='survival_pooled_' + cohort + '_scientific_acceptance', passed=True,
        inference_performed=False, provenance=provenance(),
        validated_request_count=6 if cohort == 'pilot' else 96,
        validated_unit_count=12 if cohort == 'pilot' else 1800,
        inventory=dict(path='bounded publication fixture; no scientific replay', sha256='e' * 64))


class SurvivalPooledAcceptancePublicationTests(unittest.TestCase):
    def setUp(self):
        gate._ordinary_path(SCRATCH, exists=False)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=SCRATCH)))
        self.proof_root = self.root / 'reports'
        self.stack.enter_context(mock.patch.object(gate, 'PROOF_ROOT', self.proof_root))
        self.current = self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
        self.legacy_writer = self.stack.enter_context(mock.patch.object(infrastructure, 'create_bytes',
            side_effect=AssertionError('model-artifact writer cannot publish reports')))

    def pending_path(self, cohort='pilot'):
        output = gate.default_acceptance_path(provenance(), cohort)
        return output.with_name('.' + output.name + '.pending')

    def test_default_pilot_and_formal_paths_belong_to_durable_reports(self):
        with mock.patch.object(gate, 'PROOF_ROOT', REPORT_ROOT):
            for cohort in ('pilot', 'formal'):
                output = gate.default_acceptance_path(provenance(), cohort)
                self.assertEqual(output, REPORT_ROOT / (cohort + '-' + 'd' * 64 + '.acceptance.json'))
                self.assertNotIn(infrastructure.OUTPUT_ROOT, output.parents)
        self.assertFalse(self.proof_root.exists())

    def test_pilot_and_formal_publish_complete_canonical_json_with_fsync_and_hardlink(self):
        original_fsync, original_link = os.fsync, os.link
        with mock.patch.object(gate.os, 'fsync', wraps=original_fsync) as flush, \
                mock.patch.object(gate.os, 'link', wraps=original_link) as link:
            for cohort in ('pilot', 'formal'):
                value = proof(cohort)
                output = gate.publish_acceptance(value, cohort)
                self.assertEqual(output, gate.default_acceptance_path(provenance(), cohort))
                expected = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
                self.assertEqual(output.read_bytes(), expected)
                self.assertEqual(json.loads(output.read_text()), value)
                self.assertFalse(self.pending_path(cohort).exists())
        self.assertEqual(flush.call_count, 2)
        self.assertEqual(link.call_count, 2)
        self.legacy_writer.assert_not_called()

    def test_existing_final_proof_is_preserved_and_never_overwritten(self):
        output = gate.publish_acceptance(proof())
        before = output.read_bytes()
        changed = proof()
        changed['validated_unit_count'] = 999
        with self.assertRaisesRegex(ValueError, 'preserving existing'):
            gate.publish_acceptance(changed)
        self.assertEqual(output.read_bytes(), before)
        self.assertFalse(self.pending_path().exists())

    def test_existing_pending_proof_is_preserved_without_publishing(self):
        self.proof_root.mkdir()
        pending = self.pending_path()
        pending.write_bytes(b'preserve original partial proof')
        with self.assertRaisesRegex(ValueError, 'preserving existing'):
            gate.publish_acceptance(proof())
        self.assertEqual(pending.read_bytes(), b'preserve original partial proof')
        self.assertFalse(gate.default_acceptance_path().exists())

    def test_symlink_proof_root_is_rejected_before_any_write(self):
        real_root = self.root / 'real-reports'
        real_root.mkdir()
        self.proof_root.symlink_to(real_root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            gate.publish_acceptance(proof())
        self.assertEqual(list(real_root.iterdir()), [])

    def test_dirty_source_is_rejected_even_when_proof_claims_same_dirty_source(self):
        dirty = dict(provenance(), git_dirty='1')
        value = proof()
        value['provenance'] = dirty
        self.current.return_value = dirty
        with self.assertRaisesRegex(ValueError, 'clean source'):
            gate.publish_acceptance(value)
        self.assertFalse(self.proof_root.exists())

    def test_wrong_source_pass_flag_or_cohort_scope_is_rejected_before_write(self):
        values = []
        changed = proof()
        changed['provenance']['source_tree_sha256'] = 'f' * 64
        values.append(changed)
        changed = proof()
        changed['passed'] = False
        values.append(changed)
        changed = proof('formal')
        values.append(changed)
        for value in values:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'matching scientific scope'):
                gate.publish_acceptance(value)
        self.assertFalse(self.proof_root.exists())

    def test_source_drift_before_staging_does_not_create_partial_proof(self):
        self.current.side_effect = [provenance(), dict(provenance(), source_tree_sha256='f' * 64)]
        with self.assertRaisesRegex(ValueError, 'changed before acceptance'):
            gate.publish_acceptance(proof())
        self.assertFalse(self.pending_path().exists())
        self.assertFalse(gate.default_acceptance_path(provenance()).exists())

    def test_source_drift_after_fsync_preserves_pending_without_final_publication(self):
        original_fsync = os.fsync

        def drift_after_flush(fd):
            original_fsync(fd)
            self.current.return_value = dict(provenance(), source_tree_sha256='f' * 64)

        with mock.patch.object(gate.os, 'fsync', side_effect=drift_after_flush), \
                mock.patch.object(gate.os, 'link', wraps=os.link) as link:
            with self.assertRaisesRegex(ValueError, 'partial proof preserved'):
                gate.publish_acceptance(proof())
        pending = self.pending_path()
        self.assertEqual(json.loads(pending.read_text()), proof())
        self.assertFalse(gate.default_acceptance_path(provenance()).exists())
        link.assert_not_called()

    def test_source_drift_after_hardlink_preserves_final_proof_for_review(self):
        original_link = os.link

        def drift_after_link(source, target):
            original_link(source, target)
            self.current.return_value = dict(provenance(), source_tree_sha256='f' * 64)

        with mock.patch.object(gate.os, 'link', side_effect=drift_after_link):
            with self.assertRaisesRegex(ValueError, 'preserve proof for review'):
                gate.publish_acceptance(proof())
        output = gate.default_acceptance_path(provenance())
        self.assertEqual(json.loads(output.read_text()), proof())
        self.assertFalse(self.pending_path().exists())

    def test_cli_collects_then_delegates_publication_for_both_cohorts(self):
        for cohort in ('pilot', 'formal'):
            with self.subTest(cohort=cohort):
                inventory = self.root / (cohort + '.inventory.json')
                value = proof(cohort)
                output = gate.default_acceptance_path(provenance(), cohort)
                events = []
                with mock.patch.object(gate, 'collect_acceptance',
                        side_effect=lambda path, kind: events.append(('collect', path, kind)) or value) as collect, \
                        mock.patch.object(gate, 'publish_acceptance',
                        side_effect=lambda evidence, kind: events.append(('publish', evidence, kind)) or output) as publish, \
                        redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(cli.main(['--inventory', str(inventory), '--cohort', cohort]), 0)
                collect.assert_called_once_with(inventory, cohort)
                publish.assert_called_once_with(value, cohort)
                self.assertEqual(events[0][0], 'collect')
                self.assertEqual(events[1][0], 'publish')
                self.assertEqual(json.loads(stdout.getvalue()), dict(acceptance=str(output),
                    validated_units=value['validated_unit_count'], submitted=False))
        self.legacy_writer.assert_not_called()
        self.assertFalse(self.proof_root.exists())

    def test_cli_default_cohort_and_failed_collection_never_publish(self):
        inventory = self.root / 'pilot.inventory.json'
        with mock.patch.object(gate, 'collect_acceptance', side_effect=ValueError('invalid completed evidence')) as collect, \
                mock.patch.object(gate, 'publish_acceptance') as publish:
            with self.assertRaisesRegex(ValueError, 'invalid completed'):
                cli.main(['--inventory', str(inventory)])
        collect.assert_called_once_with(inventory, 'pilot')
        publish.assert_not_called()
        self.assertFalse(self.proof_root.exists())


if __name__ == '__main__':
    unittest.main()
