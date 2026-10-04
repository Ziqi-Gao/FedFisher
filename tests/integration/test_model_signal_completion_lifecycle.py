"""Reviewed epoch hooks preserve source, prerequisite and partial-output guards."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.fedfisher_timing import artifacts as lifecycle
from effect_modifier_experiment.model_signal_ablation import completion, completion_protocol as protocol
from effect_modifier_experiment.model_signal_ablation import completion_acceptance as gate, recovery
from tests.integration.test_model_signal_recovery_lifecycle import provenance, execution


class CompletionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        scratch=Path('/scr/del6500/FedFisher/model-signal-completion-validation/lifecycle')
        scratch.mkdir(parents=True,exist_ok=True)
        self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.prov=provenance()
        self.stack.enter_context(mock.patch.object(lifecycle,'current_provenance',return_value=self.prov))
        self.parameters=dict(setting='weak_signal_05',cell='fedavg_shallow_e1_patience',variant='site_gaussian_local_permutation',shard_index=9,shard_count=16)
        self.config=protocol.run_config(protocol.FORMAL_TASK,self.parameters)
        self.prerequisites=dict(closeout=protocol.SOURCE_CLOSEOUT,pilot=dict(path=str(self.root/'pilot.json'),sha256='b'*64))

    def test_valid_formal_patience_group_keeps_partial_outputs_immutable(self):
        store=completion.CompletionStore(self.config,self.prov,self.prerequisites,self.root/self.config['run_id'])
        store.initialize();store.start(9,'iid',execution())
        started=store.cell_dir(9,'iid')/'started.json';before=started.read_bytes()
        with self.assertRaises(ValueError):store.status(9,'iid')
        with self.assertRaises(ValueError):store.start(9,'iid',execution(2))
        self.assertEqual(started.read_bytes(),before)

    def test_wrong_science_or_missing_current_pilot_binding_fails_closed(self):
        for change in ('budget','signal'):
            config=deepcopy(self.config)
            if change=='budget':config['method']['optimizer']['max_rounds']=201
            else:config['dgp']['signal_strength']=1.0
            with self.subTest(change=change),self.assertRaises(ValueError):
                completion.CompletionStore(config,self.prov,self.prerequisites,self.root/config['run_id'])
        with self.assertRaisesRegex(ValueError,'pilot prerequisite'):
            completion.CompletionStore(self.config,self.prov,dict(self.prerequisites,pilot=None),self.root/self.config['run_id'])

    def test_formal_namespace_rejects_seeds_outside_its_replacement_shards(self):
        store=completion.CompletionStore(self.config,self.prov,self.prerequisites,self.root/self.config['run_id'])
        self.assertEqual(store.identity(9,'iid')['seed'],9)
        with self.assertRaisesRegex(ValueError,'outside the fixed completion'):
            store.identity(0,'iid')
        p=dict(self.parameters,cell=protocol.BUDGET_CELL,variant='site_gaussian_shift',shard_index=0)
        config=protocol.run_config(protocol.FORMAL_TASK,p)
        store=completion.CompletionStore(config,self.prov,self.prerequisites,self.root/config['run_id'])
        self.assertEqual(store.identity(16,'noniid')['seed'],16)
        with self.assertRaisesRegex(ValueError,'outside the fixed completion'):
            store.identity(1,'noniid')

    def test_only_the_exact_archive_binding_is_allowed_and_v2_stays_restricted(self):
        for binding in (dict(protocol.SOURCE_CLOSEOUT, sha256='f'*64),
                        dict(protocol.SOURCE_CLOSEOUT, path='/data/del6500/FedFisher/archives/other/closeout.json')):
            with self.subTest(binding=binding), self.assertRaisesRegex(ValueError,'fixed immutable archive'):
                completion.CompletionStore(self.config,self.prov,
                    dict(self.prerequisites,closeout=binding),self.root/self.config['run_id'])
        with self.assertRaisesRegex(ValueError,'outside project reports'):
            recovery.validate_prerequisites(self.prerequisites,recovery.protocol.FORMAL_TASK)

    def test_historical_anchor_and_failed_new_pilot_stop_before_fitting_or_store_creation(self):
        job=dict(task=protocol.FORMAL_TASK,parameters=self.parameters,**execution())
        with mock.patch.object(recovery,'current_provenance',return_value=self.prov), \
                mock.patch.object(gate,'verify_closeout',return_value={'binding':protocol.SOURCE_CLOSEOUT}), \
                mock.patch.object(gate,'verify_pilot_acceptance',side_effect=ValueError('fresh pilot gate refused')), \
                mock.patch.object(recovery,'compute_cell') as compute, \
                mock.patch.object(completion,'CompletionStore') as store:
            with self.assertRaisesRegex(ValueError,'fresh pilot'):
                completion.run_shard(job)
            compute.assert_not_called();store.assert_not_called()
