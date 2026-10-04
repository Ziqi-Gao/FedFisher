"""Bounded saved-evidence fixtures for the post-refactor replacement epoch."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure.requests import encoded
from effect_modifier_experiment.model_signal_ablation import completion_acceptance as gate
from effect_modifier_experiment.model_signal_ablation import completion_archive as archive
from effect_modifier_experiment.model_signal_ablation import completion_protocol as protocol
from effect_modifier_experiment.model_signal_ablation import completion_requests as plan
from effect_modifier_experiment.model_signal_ablation import completion, recovery
from tests.integration.test_model_signal_recovery_acceptance import _result, _receipt, _provenance, _write, _hash


class CompletionEvidenceTests(unittest.TestCase):
    def setUp(self):
        root = Path('/scr/del6500/FedFisher/model-signal-validation/completion-evidence')
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_closeout_cannot_be_replaced_by_unbound_passed_boolean(self):
        path = self.root / 'closeout.json'
        _write(path, dict(source_release_passed=True, existing_results_archived=True))
        with self.assertRaisesRegex(ValueError, 'fixed immutable'):
            archive.verify_closeout(path)
        with mock.patch.object(archive, 'SOURCE_CLOSEOUT', dict(path=str(path), sha256='0'*64)):
            with self.assertRaisesRegex(ValueError, 'fixed immutable'):
                archive.verify_closeout(path)

    def test_copied_evidence_retains_old_identity_and_rejects_late_hash_drift(self):
        original = self.root / 'original' / 'proof.json'
        _write(original, {'source': 'original-provenance'})
        binding = gate._binding(original)
        root = self.root / 'sealed'
        copied = root / 'evidence/originals' / str(original).lstrip('/')
        _write(copied, {'source': 'original-provenance'})
        original.unlink()
        reader = archive.SealedReader(root)
        self.assertEqual(reader.read(binding), {'source': 'original-provenance'})
        _write(copied, {'source': 'new-provenance'})
        with self.assertRaisesRegex(ValueError, 'changed during'):
            reader.recheck()
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            archive.SealedReader(root).read(binding)

    def test_archived_audit_ranges_are_rechecked_after_first_producer_read(self):
        reader = archive.SealedReader(self.root)
        jid = 'fedfisher-ms-fixture'
        events = {'attempt_started': dict(schema_version=2, job_id=jid, event='job_attempt_started', attempt=1),
            'running_transition': dict(schema_version=2, job_id=jid, event='job_transition',
                project='FedFisher', source='pending', destination='running')}
        path = self.root/'audit-copy.jsonl'
        audits, index, payload = {}, {}, b''
        for event, record in events.items():
            line = (json.dumps(record)+'\n').encode()
            offset, length = len(payload), len(line)
            binding = dict(path='/data/del6500/ServerScheduler/audit/events.jsonl', offset=offset+1000,
                length=length, sha256=hashlib.sha256(line).hexdigest(), record=record)
            audits[event] = binding
            index[binding['path'], binding['offset'], length] = dict(binding,
                archive_offset=offset, archive_relative_path=path.name)
            payload += line
        path.write_bytes(payload)
        metadata = self.root/'retry.json'
        _write(metadata, {'attempts_started': 1})
        reader.audit_index = index
        reader.producer(dict(execution_receipt=_receipt(jid), producer_attempt=1, audit=audits,
            retry=gate._binding(metadata), logs={}))
        path.write_bytes(payload.replace(b'pending', b'running', 1))
        with self.assertRaisesRegex(ValueError, 'audit changed'):
            reader.recheck()

    def test_science_guard_rejects_changed_calculation_and_import_bytes(self):
        # The protected inventory comes from Git; mock its bytes, never mutate canonical science.
        fixed = ['data.py', 'models.py', 'train_model.py', 'run_one_shot_algs.py',
            'effect_modifier_experiment/dslassocov_benchmark.py',
            'effect_modifier_experiment/dslassocov_run_contract.py',
            'effect_modifier_experiment/run_dslassocov_effect_modifier.py',
            'effect_modifier_experiment/one_shot_lasso_benchmark.py',
            'effect_modifier_experiment/model_signal_ablation/protocol.py',
            'effect_modifier_experiment/model_signal_ablation/recovery_protocol.py',
            'effect_modifier_experiment/model_signal_ablation/artifacts.py',
            'effect_modifier_experiment/model_signal_ablation/runner.py']
        for name in fixed:
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original\n')
        git = mock.Mock(stdout=('\n'.join(fixed)+'\n').encode())
        with mock.patch.object(archive, 'PROJECT_ROOT', self.root), \
             mock.patch.object(archive.subprocess, 'run', return_value=git), \
             mock.patch.object(archive, '_git_blobs', return_value={name:b'original\n' for name in fixed}):
            self.assertEqual(archive.verify_science_sources()['file_count'], len(fixed))
            (self.root/'data.py').write_bytes(b'from other_science import generate\n')
            with self.assertRaisesRegex(ValueError, 'protected scientific source'):
                archive.verify_science_sources()

    def test_retained_reader_preserves_producer_and_rejects_audit_only_and_rehashed_drift(self):
        jid = 'fedfisher-ms-old-completed'
        key = ('weak_signal_05', protocol.BUDGET_CELL, 'control', 0, 'iid')
        directory = self.root/'old-output'/'seed_000_iid'
        identity = dict(run_id='original-v2', seed=0, split='iid', provenance_sha256='a'*64)
        manifest = directory.parent/'run_manifest.env'
        _write(manifest,dict(source='original-v2'))
        _write(directory/'started.json',dict(identity=identity,execution=_receipt(jid)))
        _write(directory/'result.json',dict(_result(key),identity=identity))
        (directory/'model.npz').write_bytes(b'original saved array bytes')
        _write(directory/'done.json',dict(identity=identity,
            artifacts={name:_hash(directory/name) for name in recovery.COMPLETE_FILES[:-1]}))
        row = dict(outcome='validated',final_central_state='completed',performance_accepted=True,
            manifest=gate._binding(manifest), identity=identity, job_id=jid,
            artifacts=[gate._binding(directory/name) for name in recovery.COMPLETE_FILES])
        producer = dict(execution_receipt=_receipt(jid),central_completed=True,final_state='completed')
        reader = archive.SealedReader(self.root/'archive')
        with mock.patch.object(reader,'producer') as check:
            restored = archive._retained_record(reader,row,producer)
            self.assertEqual(restored['outcome'],'validated')
            check.assert_called_once_with(producer)
            with self.assertRaisesRegex(ValueError,'classification'):
                archive._retained_record(reader,dict(row,performance_accepted=False),producer)
            with self.assertRaisesRegex(ValueError,'producer mismatch'):
                archive._retained_record(reader,row,dict(producer,central_completed=False))
        (directory/'error.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'layout changed during'):
            reader.recheck()
        (directory/'error.json').unlink()
        (directory/'model.npz').write_bytes(b'changed array and rehashed attacker marker')
        marker = gate._read_json(directory/'done.json')
        marker['artifacts']['model.npz'] = _hash(directory/'model.npz')
        _write(directory/'done.json',marker)
        with self.assertRaisesRegex(ValueError,'bytes changed'):
            archive._retained_record(archive.SealedReader(self.root/'archive'),row,producer)

    def test_retained_noncompletion_has_no_performance_or_done_promotion(self):
        jid = 'fedfisher-ms-old-cap'
        directory = self.root/'capped'/'seed_025_noniid'
        identity = dict(run_id='original-v2',seed=25,split='noniid',provenance_sha256='a'*64)
        manifest = directory.parent/'run_manifest.env'
        _write(manifest,dict(source='original-v2'))
        result = _result(('small_mlp','fedavg_small_e1_patience','site_gaussian_shift',25,'noniid'))
        result.pop('evaluation')
        result.update(status='scientific_noncompletion',identity=identity)
        _write(directory/'started.json',dict(identity=identity,execution=_receipt(jid)))
        _write(directory/'noncompletion.json',result)
        (directory/'diagnostic_model.npz').write_bytes(b'original diagnostic bytes')
        _write(directory/'noncompletion_marker.json',dict(identity=identity,outcome='validation_patience_limit_200',
            artifacts={name:_hash(directory/name) for name in recovery.NONCOMPLETE_FILES[:-1]}))
        row = dict(outcome='noncompleted',final_central_state='completed',performance_accepted=False,
            manifest=gate._binding(manifest),identity=identity,job_id=jid,
            artifacts=[gate._binding(directory/name) for name in recovery.NONCOMPLETE_FILES])
        reader = archive.SealedReader(self.root/'archive')
        producer = dict(execution_receipt=_receipt(jid),central_completed=True,final_state='completed')
        with mock.patch.object(reader,'producer'):
            restored = archive._retained_record(reader,row,producer)
            self.assertNotIn('evaluation',restored)
            self.assertEqual(restored['outcome'],'noncompleted')
            with self.assertRaisesRegex(ValueError,'classification'):
                archive._retained_record(reader,dict(row,performance_accepted=True),producer)
        _write(directory/'done.json',dict(passed=True))
        with self.assertRaisesRegex(ValueError,'layout changed'):
            archive._retained_record(reader,row,producer)


class CompletionCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v2 = gate.original
        cls.expected = {(setting, cell, variant, seed, split)
            for setting, recipe in v2.SETTINGS.items() for cell in recipe['cells']
            for variant in v2.VARIANTS for seed in range(300) for split in v2.SPLITS}
        cls.replacement_keys = {(p['setting'], p['cell'], p['variant'], seed, split)
            for p in protocol.FORMAL_PARAMETERS for seed in protocol.seed_sequence(protocol.FORMAL_TASK, p)
            for split in protocol.SPLITS}
        cls.replacement = {key: dict(outcome='validated') for key in cls.replacement_keys}
        caps = {(setting, 'fedavg_small_e1_patience', variant, 25, 'noniid')
                for setting in v2.SETTINGS for variant in ('site_gaussian_shift', 'site_gaussian_local_permutation')}
        cls.retained = {key: dict(outcome='noncompleted' if key in caps else 'validated')
                        for key in cls.expected-cls.replacement_keys}

    def test_exact_whole_failed_shards_and_separate_six_caps_keep_300_denominators(self):
        proof = gate.validate_composite_coverage(self.retained, self.replacement, self.replacement_keys)
        self.assertEqual(proof['scientific_units'], 172800)
        self.assertEqual(proof['completed_units'], 172794)
        self.assertEqual(proof['noncompleted_units'], 6)
        self.assertFalse(proof['whole_cohort_scientific_success'])
        self.assertTrue(all(row['planned'] == 300 for row in proof['conditions']))

    def test_failed_shard_partial_promotion_gap_and_unknown_terminal_fail_closed(self):
        key = next(iter(self.replacement))
        reduced = dict(self.replacement)
        reduced.pop(key)
        with self.assertRaisesRegex(ValueError, 'gaps, duplicates'):
            gate.validate_composite_coverage(self.retained, reduced, self.replacement_keys)
        promoted = dict(self.retained, **{})
        promoted[key] = self.replacement[key]
        with self.assertRaisesRegex(ValueError, 'gaps, duplicates'):
            gate.validate_composite_coverage(promoted, self.replacement, self.replacement_keys)
        changed = dict(self.replacement)
        changed[key] = dict(outcome='unknown_failure')
        with self.assertRaisesRegex(ValueError, 'unknown terminal'):
            gate.validate_composite_coverage(self.retained, changed, self.replacement_keys)
        capped = next(k for k,v in self.retained.items() if v['outcome']=='noncompleted')
        promoted = dict(self.retained)
        promoted[capped] = dict(outcome='validated')
        with self.assertRaisesRegex(ValueError, 'six historical'):
            gate.validate_composite_coverage(promoted, self.replacement, self.replacement_keys)


class CompletionCollectorTests(unittest.TestCase):
    """Exercise all 16 real request/receipt/hash/summary/producer/parity gates.

    Model-array semantic validation is an explicit fixture boundary, covered by
    the real tiny store tests. These fixtures never fit a model or create data.
    """
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/model-signal-validation/completion-collector')
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        for name, relative in dict(OUTBOX_ROOT='outbox', CENTRAL_JOBS='central/jobs',
                CENTRAL_LOGS='central/logs', CENTRAL_RETRIES='central/retries', CENTRAL_AUDIT='central/audit.jsonl').items():
            self.stack.enter_context(mock.patch.object(gate.old, name, self.root/relative))
        self.stack.enter_context(mock.patch.object(gate.previous, 'REPORT_ROOT', self.root/'reports'))
        self.stack.enter_context(mock.patch.object(gate, 'PROOF_ROOT', self.root/'proofs'))
        self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=_provenance()))
        guard = dict(commit='reviewed', file_count=12, files_sha256='a'*64)
        self.stack.enter_context(mock.patch.object(archive, 'verify_science_sources', return_value=guard))
        source = self.root/'source.py'
        source.write_text('# fixture\n')
        self.sources = {'source':gate._binding(source)}
        self.stack.enter_context(mock.patch.object(plan, 'review_sources', return_value=self.sources))
        owner = self

        class FixtureStore:
            execution_protocol = protocol
            def __init__(self, config, provenance, prerequisites):
                self.config, self.provenance, self.prerequisites = config, provenance, prerequisites
                self.run_id, self.config_sha256 = config['run_id'], gate.digest(config)
                self.directory = owner.root/'outputs'/self.run_id
            def _current(self):
                owner.assertEqual(self.provenance, _provenance())
            def _manifest(self):
                owner.assertEqual(gate._read_json(self.directory/'run_manifest.env'), self.config)
            def identity(self, seed, split):
                return dict(run_id=self.run_id, config_sha256=self.config_sha256,
                    provenance_sha256=self.provenance['provenance_sha256'], seed=seed, split=split,
                    cell=self.config['cell'], variant=self.config['variant'])
            def cell_dir(self, seed, split):
                return self.directory/('seed_%03d_%s' % (seed,split))
            def _started(self, seed, split):
                row = gate._read_json(self.cell_dir(seed,split)/'started.json')
                if row['identity'] != self.identity(seed,split) or row['prerequisites'] != self.prerequisites:
                    raise ValueError('fixture started identity/prerequisites mismatch')
                gate.old.validate_execution(row['execution'])
                return row
            def _read(self, seed, split):
                return gate._read_json(self.cell_dir(seed,split)/'result.json'), {}
            def _read_noncompletion(self, seed, split):
                return gate._read_json(self.cell_dir(seed,split)/'noncompletion.json'), {}

        self.stack.enter_context(mock.patch.object(completion, 'CompletionStore', FixtureStore))
        self.requests, self.directories, self.references, audit = [], {}, {}, []
        for spec in plan.build_requests():
            request, config = spec['request'], spec['configuration']
            jid, p = request['job_id'], request['parameters']
            path = gate.old.OUTBOX_ROOT/spec['filename']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded(request))
            self.requests.append(dict(path=str(path), sha256=_hash(path), job_id=jid,
                task=request['task'], parameters=p, run_id=spec['run_id'], configuration=config,
                configuration_sha256=spec['configuration_sha256'], seeds=spec['seed_sequence'], splits=spec['splits']))
            prerequisites = dict(closeout=protocol.SOURCE_CLOSEOUT, pilot=None)
            store = FixtureStore(config, _provenance(), prerequisites)
            _write(store.directory/'run_manifest.env', config)
            terminals = []
            for split in protocol.SPLITS:
                key = p['setting'],p['cell'],p['variant'],0,split
                result = _result(key)
                self.references[key] = deepcopy(result)
                directory = store.cell_dir(0,split)
                self.directories[key] = directory
                _write(directory/'started.json', dict(identity=store.identity(0,split), execution=_receipt(jid), prerequisites=prerequisites))
                _write(directory/'result.json', dict(result, identity=store.identity(0,split)))
                (directory/'model.npz').write_bytes(b'semantic-model-fixture-boundary')
                _write(directory/'done.json', dict(identity=store.identity(0,split),
                    artifacts={name:_hash(directory/name) for name in recovery.COMPLETE_FILES[:-1]}))
                terminals.append(dict(seed=0, split=split, outcome='complete',
                    artifacts={name:_hash(directory/name) for name in recovery.COMPLETE_FILES}))
            execution = _receipt(jid)
            job = dict(request, state='completed', exit_code=0,
                **{k:v for k,v in execution.items() if k not in ('job_id','attempt')})
            for kind in ('stdout','stderr'):
                path = gate.old.CENTRAL_LOGS/(jid+'.attempt-001.'+kind+'.log')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('')
                job[kind+'_log'] = str(path)
            _write(gate.old.CENTRAL_JOBS/(jid+'.json'), job)
            _write(gate.old.CENTRAL_RETRIES/(jid+'.json'), dict(schema_version=2, job_id=jid,
                attempts_started=1, max_attempts=3, next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
            _write(store.directory/'shard_summaries'/(jid+'.json'), recovery._summary_body(store,job,execution,verified_units=terminals))
            audit.extend([dict(schema_version=2,job_id=jid,event='job_attempt_started',attempt=1),
                dict(schema_version=2,job_id=jid,event='job_transition',project='FedFisher',source='pending',destination='running')])
        gate.old.CENTRAL_AUDIT.write_text(''.join(json.dumps(row)+'\n' for row in audit))
        closeout = dict(binding=protocol.SOURCE_CLOSEOUT, pilot_references=self.references,
            pilot_reference_binding=dict(path=str(self.root/'original-proof.json'),sha256='e'*64),
            source_guard=guard, reader=mock.Mock())
        self.stack.enter_context(mock.patch.object(gate,'verify_closeout',return_value=closeout))
        self.stack.enter_context(mock.patch.object(archive,'reference_result',side_effect=lambda closeout,key:self.references[key]))
        self.inventory_path = gate.previous.REPORT_ROOT/'pilot.inventory.json'
        _write(self.inventory_path,dict(schema_version=1,bundle_ready=True,generation=1,
            provenance=_provenance(),task=protocol.PILOT_TASK,closeout=protocol.SOURCE_CLOSEOUT,
            review_sources=self.sources,request_count=8,scientific_cell_count=16,requests=self.requests))

    def _rehash(self, directory):
        marker = gate._read_json(directory/'done.json')
        marker['artifacts'] = {name:_hash(directory/name) for name in recovery.COMPLETE_FILES[:-1]}
        _write(directory/'done.json',marker)
        path = next((directory.parent/'shard_summaries').iterdir())
        summary = gate._read_json(path)
        for row in summary['units']:
            if directory.name == 'seed_%03d_%s' % (row['seed'],row['split']):
                row['artifacts'] = {name:_hash(directory/name) for name in recovery.COMPLETE_FILES}
        _write(path,summary)

    def test_full_sixteen_unit_collector_and_fresh_proof_replay(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        self.assertEqual((proof['validated_request_count'],proof['validated_unit_count']),(8,16))
        self.assertEqual(proof['checks']['exact_success_parity_units'],16)
        self.assertEqual(proof['checks']['terminal_training_pairs']['terminal_training_pairs'],8)
        path = gate.default_acceptance_path()
        _write(path,proof)
        self.assertEqual(gate.verify_pilot_acceptance(path),proof)
        _write(path,dict(proof,complete_units=15))
        with self.assertRaisesRegex(ValueError,'freshly verified'):
            gate.verify_pilot_acceptance(path)

    def test_rehashed_model_numerical_tampering_and_type_drift_fail_parity(self):
        directory = next(iter(self.directories.values()))
        result = gate._read_json(directory/'result.json')
        result['fit']['numeric_parameter'] += .01
        _write(directory/'result.json',result)
        self._rehash(directory)
        with self.assertRaisesRegex(ValueError,'numeric fitting path'):
            gate.collect_pilot_acceptance(self.inventory_path)
        reference = deepcopy(next(iter(self.references.values())))
        result = deepcopy(reference)
        reference['fit']['numeric_parameter'] = 1.0
        result['fit']['numeric_parameter'] = 1
        with self.assertRaisesRegex(ValueError,'numeric fitting path'):
            gate.previous.validate_numeric_parity(result,reference)

    def test_source_drift_schema_and_partial_files_fail_closed(self):
        with mock.patch.object(gate,'current_provenance',side_effect=[_provenance(),dict(_provenance(),git_dirty='1')]):
            with self.assertRaisesRegex(ValueError,'source changed'):
                gate.collect_pilot_acceptance(self.inventory_path)
        inventory = gate._read_json(self.inventory_path)
        _write(self.inventory_path,dict(inventory,schema_version=2))
        with self.assertRaisesRegex(ValueError,'reviewed source'):
            gate.collect_pilot_acceptance(self.inventory_path)
        _write(self.inventory_path,inventory)
        directory = next(iter(self.directories.values()))
        (directory/'failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'partial or mixed'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_central_failure_and_forged_summary_writer_cannot_supply_success(self):
        request = self.requests[0]
        path = gate.old.CENTRAL_JOBS/(request['job_id']+'.json')
        job = gate._read_json(path)
        _write(path,dict(job,state='failed',exit_code=-15))
        with self.assertRaisesRegex(ValueError,'completed central job'):
            gate.collect_pilot_acceptance(self.inventory_path)
        _write(path,job)
        directory = next(iter(self.directories.values()))
        path = next((directory.parent/'shard_summaries').iterdir())
        summary = gate._read_json(path)
        summary['producer_execution']['estimated_runtime_seconds'] = 21.
        _write(path,summary)
        with self.assertRaisesRegex(ValueError,'producing-attempt receipt'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_previous_success_cannot_be_replaced_by_a_verified_or_unknown_cap(self):
        records = {key:dict(outcome='validated',result=value) for key,value in self.references.items()}
        key = next(iter(records))
        for outcome in ('noncompleted','unknown_failure'):
            records[key] = dict(outcome=outcome,result=self.references[key])
            with self.assertRaisesRegex(ValueError,'became a noncompletion'):
                gate.validate_pilot_observations(records,self.references)


if __name__ == '__main__':
    unittest.main()
