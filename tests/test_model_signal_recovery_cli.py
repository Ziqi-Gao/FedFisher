"""Immutable publication and source-change checks around the saved-evidence gate."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure.artifacts import safe_path
from effect_modifier_experiment.model_signal_ablation import recovery_acceptance as gate
from scripts import validate_model_signal_recovery as cli


class RecoveryAcceptanceCLITests(unittest.TestCase):
    def setUp(self):
        parent = safe_path(Path('/scr/del6500/FedFisher/model-signal-recovery-validation/cli'))
        parent.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=parent)))
        self.provenance = dict(git_dirty='0', provenance_sha256='a' * 64)
        self.current = self.stack.enter_context(mock.patch.object(gate, 'current_provenance',
                                                                  return_value=self.provenance))
        self.stack.enter_context(mock.patch.object(gate, 'PROOF_ROOT', self.root))
        self.proof = dict(provenance=self.provenance, validated_request_count=290,
                          validated_unit_count=588, formal_eligible=True)
        self.collect = self.stack.enter_context(mock.patch.object(gate, 'collect_pilot_acceptance',
                                                                 return_value=self.proof))
        self.output = gate.default_acceptance_path(self.provenance)
        self.args = ['--inventory', str(self.root / 'fixture.inventory.json')]

    def invoke(self):
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(cli.main(self.args), 0)
        return json.loads(stream.getvalue())

    def test_first_publication_and_read_only_revalidation_do_not_overwrite(self):
        first = self.invoke()
        before = self.output.read_bytes()
        second = self.invoke()
        self.assertTrue(first['published'])
        self.assertFalse(second['published'])
        self.assertFalse(first['submitted'])
        self.assertEqual(first['sha256'], second['sha256'])
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self.collect.call_count, 2)

    def test_changed_fresh_evidence_preserves_the_existing_proof(self):
        self.invoke()
        before = self.output.read_bytes()
        changed = deepcopy(self.proof)
        changed['validated_unit_count'] = 587
        self.collect.return_value = changed
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.invoke()
        self.assertEqual(self.output.read_bytes(), before)

    def test_source_drift_after_pending_write_preserves_partial_proof(self):
        self.current.side_effect = [self.provenance, self.provenance,
                                   dict(self.provenance, provenance_sha256='f' * 64)]
        with self.assertRaisesRegex(ValueError, 'source changed while publishing'):
            self.invoke()
        self.assertFalse(self.output.exists())
        pending = self.output.with_name('.' + self.output.name + '.pending')
        self.assertTrue(pending.exists())
        self.current.side_effect = None
        with self.assertRaisesRegex(FileExistsError, 'partial'):
            self.invoke()


if __name__ == '__main__':
    unittest.main()
