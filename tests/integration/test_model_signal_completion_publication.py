"""New-source completion cannot publish formal work without its own real pilot."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests.integration.request_fixtures import bind_review_sources
from tests.integration.test_model_signal_recovery_requests import provenance
from effect_modifier_experiment.scheduling.publication import model_signal_completion as publisher
from effect_modifier_experiment.model_signal_ablation import completion_acceptance as gate
from scripts import validate_model_signal_completion as cli


class CompletionPublicationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        scratch = Path('/scr/del6500/FedFisher/model-signal-completion-validation/publication')
        scratch.mkdir(parents=True, exist_ok=True)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.stack.enter_context(mock.patch.object(publisher, 'OUTBOX_ROOT', self.root/'outbox'))
        self.stack.enter_context(mock.patch.object(publisher, 'REPORT_ROOT', self.root/'reports'))
        bind_review_sources(self.stack, publisher)
        self.current = self.stack.enter_context(mock.patch.object(publisher, 'compute_provenance', return_value=provenance()))
        self.closeout = self.stack.enter_context(mock.patch.object(publisher, '_closeout',
            return_value={'binding': publisher.protocol.SOURCE_CLOSEOUT}))

    def test_pilot_publication_is_complete_nested_and_immutable(self):
        inventory = publisher.write_requests()
        self.assertEqual(inventory['request_count'], 8)
        self.assertEqual(inventory['scientific_cell_count'], 16)
        self.assertFalse(inventory['submitted'])
        self.assertFalse(inventory['formal_plan']['publication_authorized'])
        self.assertEqual(len(list((self.root/'outbox').glob('*.json'))), 0)
        self.assertEqual(json.loads(Path(inventory['inventory_path']).read_text()), inventory)
        self.assertEqual(self.closeout.call_count, 2)
        with self.assertRaises(FileExistsError):
            publisher.write_requests()

    def test_dirty_source_and_bad_anchor_stop_before_outbox_creation(self):
        self.current.return_value = provenance(git_dirty='1')
        with self.assertRaisesRegex(ValueError, 'dirty'):
            publisher.write_requests()
        self.current.return_value = provenance()
        self.closeout.side_effect = ValueError('bad immutable archive')
        with self.assertRaisesRegex(ValueError, 'immutable archive'):
            publisher.write_requests()
        self.assertFalse((self.root/'outbox').exists())

    def test_formal_requires_new_pilot_and_cannot_use_old_scope_or_provenance(self):
        with self.assertRaisesRegex(ValueError, 'same-source'):
            publisher.write_requests(cohort='formal')
        pilot_path = self.root/'pilot.json'
        pilot_path.write_text('{}')
        proof = dict(passed=True, formal_eligible=True, provenance=provenance(),
            closeout=publisher.protocol.SOURCE_CLOSEOUT, validated_request_count=8, validated_unit_count=16)
        with mock.patch.object(gate, 'verify_pilot_acceptance', return_value=proof):
            inventory = publisher.write_requests(cohort='formal', pilot_acceptance=pilot_path)
        self.assertEqual(inventory['request_count'], 24)
        self.assertEqual(inventory['scientific_cell_count'], 896)
        self.assertEqual(inventory['retained_noncompleted_units'], 6)
        for key, wrong in (('validated_unit_count',588), ('provenance',provenance(git_commit='f'*40)), ('passed',False)):
            changed = dict(proof, **{key:wrong})
            with mock.patch.object(gate, 'verify_pilot_acceptance', return_value=changed), self.assertRaisesRegex(ValueError,'incompatible'):
                publisher.write_requests(generation=2,cohort='formal',pilot_acceptance=pilot_path)
        self.assertFalse((self.root/'reports'/publisher.inventory_name(2,'formal')).exists())

    def test_source_and_archive_drift_preserve_partial_publication(self):
        self.current.side_effect = [provenance(),provenance(git_commit='f'*40)]
        with self.assertRaisesRegex(ValueError,'source changed'):
            publisher.write_requests()
        self.assertFalse((self.root/'reports'/publisher.inventory_name()).exists())
        self.current.side_effect = None
        self.closeout.side_effect = [dict(binding=publisher.protocol.SOURCE_CLOSEOUT), dict(binding={'path':'changed','sha256':'f'*64})]
        with self.assertRaisesRegex(ValueError,'anchor changed'):
            publisher.write_requests(generation=2)
        self.assertFalse((self.root/'reports'/publisher.inventory_name(2)).exists())

    def test_source_drift_inside_final_archive_verification_prevents_inventory(self):
        def verify_archive():
            if self.closeout.call_count == 2:
                self.current.return_value = provenance(git_commit='f'*40)
            return dict(binding=publisher.protocol.SOURCE_CLOSEOUT)
        self.closeout.side_effect = verify_archive
        with self.assertRaisesRegex(ValueError,'source changed during final'):
            publisher.write_requests()
        self.assertFalse((self.root/'reports'/publisher.inventory_name()).exists())


class CompletionProofPublicationTests(unittest.TestCase):
    def test_saved_evidence_cli_is_immutable_and_rejects_incomplete_fresh_evidence(self):
        scratch = Path('/scr/del6500/FedFisher/model-signal-completion-validation/cli')
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory, ExitStack() as stack:
            root = Path(directory); prov=provenance()
            stack.enter_context(mock.patch.object(gate,'PROOF_ROOT',root))
            stack.enter_context(mock.patch.object(gate,'current_provenance',return_value=prov))
            proof=dict(provenance=prov,validated_request_count=8,validated_unit_count=16,formal_eligible=True)
            collect=stack.enter_context(mock.patch.object(gate,'collect_pilot_acceptance',return_value=proof))
            args=['--inventory',str(root/'fixture.inventory.json')]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args),0)
                output=gate.default_acceptance_path(prov); before=output.read_bytes()
                self.assertEqual(cli.main(args),0)
                self.assertEqual(output.read_bytes(),before)
                collect.return_value=dict(proof,validated_unit_count=15)
                with self.assertRaisesRegex(ValueError,'differs'):
                    cli.main(args)
                self.assertEqual(output.read_bytes(),before)
