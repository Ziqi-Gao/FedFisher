"""Saved-evidence fixtures, with no inference, fitting or central state writes.

Only the numerical artifact-store boundary and frozen source/historical anchors
are replaced. Exact request bytes, job ownership, actual producing attempts,
retry/log/audit evidence, paired coverage and immutable proof publication remain
the production implementations. All fixture files live in FedFisher scratch.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure.requests import encoded
from effect_modifier_experiment.survival import acceptance as historical_gate
from effect_modifier_experiment.survival_pooled import acceptance as pooled_gate
from effect_modifier_experiment.survival_mlp8 import acceptance as gate, protocol
from effect_modifier_experiment.survival_mlp8.requests import build_requests


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                source_tree_sha256='c' * 64, provenance_sha256='d' * 64)


def execution(job_id, attempt=1):
    return dict(job_id=job_id, attempt=attempt, execution_profile='cpu-1core',
                estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
                gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


class SurvivalMLP8AcceptanceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-mlp8-validation/acceptance')
        gate._ordinary_path(scratch, exists=False)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        for key, suffix in (('OUTBOX_ROOT', 'outbox'), ('INVENTORY_ROOT', 'reports'), ('PROOF_ROOT', 'proofs')):
            self.stack.enter_context(mock.patch.object(gate, key, self.root / suffix))
        # Shared helper globals stay at their defining modules. Redirect paths
        # rather than replacing actual completed-job/producer/audit validation.
        self.jobs_root = self.root / 'central/jobs'
        self.stack.enter_context(mock.patch.object(pooled_gate, 'CENTRAL_JOBS', self.jobs_root))
        for key, suffix in (('CENTRAL_LOGS', 'central/logs'), ('CENTRAL_RETRIES', 'central/retries'),
                            ('CENTRAL_AUDIT', 'central/audit.jsonl')):
            self.stack.enter_context(mock.patch.object(historical_gate, key, self.root / suffix))
            if key != 'CENTRAL_AUDIT':
                self.stack.enter_context(mock.patch.object(gate, key, self.root / suffix))
        self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
        self.historical = {'fixture_acceptance': {'path': str(self.root / 'old.json'), 'sha256': '1' * 64}}
        self.stack.enter_context(mock.patch.object(gate, 'historical_binding', return_value=self.historical))
        self.validate_historical = self.stack.enter_context(mock.patch.object(gate, 'validate_historical_inputs'))
        self.sources = {'fixture': {'path': str(self.root / 'source.py'), 'sha256': '2' * 64}}
        self.stack.enter_context(mock.patch.object(gate, 'review_sources', return_value=self.sources))
        self.directories, self.status_calls, self.prerequisites = {}, [], []
        self.specs = build_requests()
        requests, audit = [], []
        for spec in self.specs:
            raw, config = spec['request'], spec['configuration']
            request_path = gate.OUTBOX_ROOT / 'bundle' / spec['filename']
            write_json(request_path, raw)
            requests.append(self.request_record(spec, request_path))
            parent = self.root / 'outputs' / config['run_id']
            write_json(parent / 'run_manifest.env', {'fixture_config': config})
            for seed in spec['seed_sequence']:
                directory = parent / ('seed_%03d' % seed)
                directory.mkdir()
                self.directories[config['scenario'], seed] = directory
                write_json(directory / 'started.json', dict(execution=execution(raw['job_id'])))
                write_json(directory / 'result.json', self.result(config['scenario'], seed))
                (directory / 'model.npz').write_bytes(b'fabricated numerical store boundary')
                self.rehash(directory)
            self.write_completed_job(raw)
            write_json(historical_gate.CENTRAL_RETRIES / (raw['job_id'] + '.json'),
                dict(schema_version=2, job_id=raw['job_id'], attempts_started=1, max_attempts=3,
                     next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
            audit.extend([dict(schema_version=2, job_id=raw['job_id'], event='job_attempt_started', attempt=1),
                dict(schema_version=2, job_id=raw['job_id'], event='job_transition', project='FedFisher',
                     source='pending', destination='running')])
        historical_gate.CENTRAL_AUDIT.write_text(''.join(json.dumps(row) + '\n' for row in audit))
        self.inventory = self.inventory_for('pilot', self.specs, requests)
        self.inventory_path = gate.INVENTORY_ROOT / 'pilot.inventory.json'
        write_json(self.inventory_path, self.inventory)
        owner = self

        class FabricatedStore:
            artifact_names = ('started.json', 'model.npz', 'result.json')
            result_name, done_name = 'result.json', 'done.json'

            def __init__(self, config, source, prerequisite=None):
                owner.assertEqual(source, provenance())
                self.config, self.directory = config, owner.root / 'outputs' / config['run_id']
                owner.prerequisites.append(prerequisite)

            def cell_dir(self, seed):
                return self.directory / ('seed_%03d' % seed)

            def status(self, seed):
                owner.status_calls.append((self.config['scenario'], seed))
                directory = self.cell_dir(seed)
                if not (directory / self.done_name).exists():
                    return 'partial'
                marker = json.loads((directory / self.done_name).read_text())
                if any(checksum(directory / name) != value for name, value in marker.items()):
                    raise ValueError('fabricated artifact checksum mismatch')
                return 'validated'

        self.stack.enter_context(mock.patch.object(gate, 'ArtifactStore', FabricatedStore))

    def request_record(self, spec, path):
        raw = spec['request']
        return dict(path=str(path), sha256=spec['request_sha256'], job_id=raw['job_id'],
            task=raw['task'], parameters=raw['parameters'], seeds=spec['seed_sequence'],
            run_id=spec['run_id'], configuration=spec['configuration'],
            configuration_sha256=spec['configuration_sha256'])

    def write_completed_job(self, raw):
        job = dict(raw, state='completed', exit_code=0, **{key: value for key, value in
                   execution(raw['job_id']).items() if key not in ('job_id', 'attempt')})
        for kind in ('stdout', 'stderr'):
            path = historical_gate.CENTRAL_LOGS / ('%s.attempt-001.%s.log' % (raw['job_id'], kind))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
            job[kind + '_log'] = str(path)
        write_json(self.jobs_root / (raw['job_id'] + '.json'), job)

    def inventory_for(self, cohort, specs, requests):
        seeds = list(protocol.PILOT_SEEDS if cohort == 'pilot' else protocol.FORMAL_SEEDS)
        return dict(schema_version=1, bundle='fedfisher-survival-mlp8-v1-' + cohort,
            generation=1, task=protocol.PILOT_TASK if cohort == 'pilot' else protocol.FORMAL_TASK,
            request_count=len(specs), scientific_cell_count=2 * len(seeds), fitted_model_count=8 * len(seeds),
            coverage=dict(scenarios=list(protocol.SCENARIOS), seeds=seeds, models=list(protocol.TAGS),
                          pseudo_scopes=list(protocol.PSEUDO_SCOPES)),
            provenance=provenance(), project_head=provenance()['git_commit'], worktree_status='clean',
            bundle_ready=True, review_sources=self.sources, historical_inputs=self.historical, requests=requests)

    def result(self, scenario, seed):
        # Poor accuracy is an observation, not an invented completion threshold.
        summary = dict(arms={'0': {'rmse': .9}, '1': {'rmse': .8}}, combined={'rmse': .85},
                       factual={'rmse': .9}, contrast={'rmse': 10.})
        return dict(status='complete', site_sizes=list(range(500, 510)),
            fingerprints=dict(training=hashlib.sha256((scenario + str(seed)).encode()).hexdigest(),
                              evaluation=hashlib.sha256(('x' + str(seed)).encode()).hexdigest(),
                              evaluation_truth=hashlib.sha256(('truth' + str(seed)).encode()).hexdigest()),
            prediction={tag: deepcopy(summary) for tag in protocol.TAGS},
            comparisons={tag: {scope: {'combined': {'rmse': .8}} for scope in protocol.PSEUDO_SCOPES}
                         for tag in protocol.FEDERATED_TAGS},
            fit={tag: dict(config={'architecture': 'mlp8', 'epochs': 30}, sample_count=5045,
                            total_fit_seconds=1., stop_reason='fixed_epoch_budget') for tag in protocol.TAGS},
            timing=dict(stages={name: {'wall_seconds': 3., 'cpu_seconds': 2.} for name in
                                ('historical_input_validation', 'data_preparation', 'fit', 'probability_evaluation')}),
            memory={'process_peak_rss_kib': 1000, 'scope': 'fabricated fixture'})

    def rehash(self, directory):
        write_json(directory / 'done.json', {name: checksum(directory / name)
                   for name in ('started.json', 'model.npz', 'result.json')})

    def change_result(self, key, mutate):
        directory = self.directories[key]
        result = json.loads((directory / 'result.json').read_text())
        mutate(result)
        write_json(directory / 'result.json', result)
        self.rehash(directory)

    def collect(self):
        return gate.collect_acceptance(self.inventory_path)

    def test_exact_four_units_sixteen_models_fresh_replay_without_inference(self):
        with mock.patch('effect_modifier_experiment.survival_mlp8.methods.MLP8Fit.predict_eta',
                        side_effect=AssertionError('acceptance must not infer')):
            proof = self.collect()
            self.assertEqual(proof['validated_request_count'], 2)
            self.assertEqual(proof['validated_unit_count'], 4)
            self.assertEqual(sum(len(unit['observations']['probability_accuracy']) for unit in proof['units']), 16)
            self.assertTrue(proof['paired_checks']['passed'])
            self.assertEqual(proof['paired_checks']['complete_seed_count'], 2)
            self.assertFalse(proof['inference_performed'])
            self.assertTrue(all(p is None for p in self.prerequisites))
            self.assertTrue(all(not unit['observations']['performance_threshold_applied'] for unit in proof['units']))
            path = gate.publish_acceptance(proof)
            self.assertEqual(gate.verify_pilot_acceptance(path), proof)
        self.assertEqual(len(self.status_calls), 8)
        self.assertGreaterEqual(self.validate_historical.call_count, 4)

    def test_actual_audit_once_per_attempt_and_recheck_each_unit(self):
        with mock.patch.object(gate, '_attempt_audit', wraps=gate._attempt_audit) as scans, \
             mock.patch.object(gate, '_recheck_producer', wraps=gate._recheck_producer) as rechecks:
            self.collect()
        self.assertEqual(scans.call_count, 2)
        self.assertEqual(rechecks.call_count, 4)
        self.assertEqual({call.args for call in scans.call_args_list},
                         {(row['job_id'], 1) for row in self.inventory['requests']})

    def test_stale_passed_boolean_and_old_namespace_cannot_open_gate(self):
        proof = self.collect()
        path = gate.default_acceptance_path()
        for changes in (dict(validated_unit_count=999), dict(scope='survival_pooled_pilot_scientific_acceptance'),
                        dict(provenance=dict(provenance(), git_commit='f' * 40))):
            write_json(path, dict(proof, **changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gate.verify_pilot_acceptance(path)
        write_json(self.inventory_path, dict(self.inventory, bundle='fedfisher-survival-pooled-v1-pilot'))
        with self.assertRaisesRegex(ValueError, 'fixed coverage or frozen source'):
            self.collect()

    def test_request_bytes_counts_and_duplicate_recipe_fail_closed(self):
        original = deepcopy(self.inventory)
        for mutate in (lambda x: x['requests'].__setitem__(-1, x['requests'][0]),
                       lambda x: x.update(request_count=1), lambda x: x.update(fitted_model_count=8)):
            changed = deepcopy(original)
            mutate(changed)
            write_json(self.inventory_path, changed)
            with self.assertRaises(ValueError):
                self.collect()
        write_json(self.inventory_path, original)
        path = Path(original['requests'][0]['path'])
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'request bytes changed'):
            self.collect()

    def test_source_coverage_historical_review_bindings_rejected(self):
        for changes in (dict(provenance=dict(provenance(), source_tree_sha256='e' * 64)),
                        dict(coverage=dict(self.inventory['coverage'], seeds=[0])),
                        dict(historical_inputs={'different': True})):
            write_json(self.inventory_path, dict(self.inventory, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'fixed coverage or frozen source'):
                self.collect()
        write_json(self.inventory_path, self.inventory)
        with mock.patch.object(gate, 'review_sources', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'reviewed sources changed'):
                self.collect()

    def test_job_owner_state_parameters_and_boolean_exit_zero_rejected(self):
        request = self.inventory['requests'][0]
        path = self.jobs_root / (request['job_id'] + '.json')
        original = json.loads(path.read_text())
        for changes in (dict(state='failed'), dict(project='Other'), dict(exit_code=False),
                        dict(task=protocol.FORMAL_TASK), dict(parameters={'scenario': 'wrong'})):
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'matching completed'):
                self.collect()

    def test_actual_producer_and_exact_attempt_audit_required(self):
        directory = self.directories['iid', 0]
        started = json.loads((directory / 'started.json').read_text())
        started['execution']['job_id'] = 'fedfisher-survival-foreign-producer'
        write_json(directory / 'started.json', started)
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError, 'another scheduler job'):
            self.collect()
        started['execution']['job_id'] = self.inventory['requests'][0]['job_id']
        write_json(directory / 'started.json', started)
        self.rehash(directory)
        audit = historical_gate.CENTRAL_AUDIT
        audit.write_text('\n'.join(line for line in audit.read_text().splitlines()
                         if self.inventory['requests'][0]['job_id'] not in line) + '\n')
        with self.assertRaisesRegex(ValueError, 'exact central start/running audit'):
            self.collect()

    def test_retry_preserves_earlier_producer_and_measured_fit_time(self):
        job_id = self.inventory['requests'][0]['job_id']
        path = self.jobs_root / (job_id + '.json')
        job = json.loads(path.read_text())
        job['estimated_runtime_seconds'] = 90.
        for kind in ('stdout', 'stderr'):
            log = historical_gate.CENTRAL_LOGS / ('%s.attempt-002.%s.log' % (job_id, kind))
            log.write_text('')
            job[kind + '_log'] = str(log)
        write_json(path, job)
        retry_path = historical_gate.CENTRAL_RETRIES / (job_id + '.json')
        retry = json.loads(retry_path.read_text())
        retry['attempts_started'] = 2
        write_json(retry_path, retry)
        proof = self.collect()
        producer = proof['units'][0]['producer_evidence']
        self.assertEqual((producer['producer_attempt'], producer['final_completed_attempt']), (1, 2))
        self.assertTrue(producer['resumed_from_earlier_attempt'])
        self.assertEqual(proof['units'][0]['measured_stage_wall_seconds'], 12.)

    def test_artifact_hash_failure_partial_unit_and_paired_truth_rejected(self):
        directory = self.directories['iid', 0]
        (directory / 'model.npz').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.collect()
        self.rehash(directory)
        done = (directory / 'done.json').read_bytes()
        (directory / 'done.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.collect()
        (directory / 'done.json').write_bytes(done)
        self.change_result(('covariate_shift', 49), lambda r: r['fingerprints'].update(evaluation_truth='e' * 64))
        with self.assertRaisesRegex(ValueError, 'independent test rows and truth'):
            self.collect()

    def test_changed_saved_summary_invalidates_previous_proof(self):
        proof = self.collect()
        path = gate.publish_acceptance(proof)
        self.change_result(('iid', 0), lambda r: r['prediction']['fedfisher_diag']['combined'].update(rmse=.12))
        with self.assertRaisesRegex(ValueError, 'fresh verification'):
            gate.verify_pilot_acceptance(path)

    def test_internally_rehashed_mutation_during_store_status_is_rejected(self):
        store_type, original_status = gate.ArtifactStore, gate.ArtifactStore.status

        def mutate_after_validation(store, seed):
            state = original_status(store, seed)
            if (store.config['scenario'], seed) == ('iid', 0):
                self.change_result(('iid', 0), lambda r:
                    r['prediction']['fedfisher_diag']['combined'].update(rmse=.12))
            return state

        with mock.patch.object(store_type, 'status', new=mutate_after_validation):
            with self.assertRaisesRegex(ValueError, 'scientific artifact changed'):
                self.collect()

    def test_final_source_and_review_drift_after_producer_rechecks_rejected(self):
        changed = dict(provenance(), source_tree_sha256='e' * 64)
        with mock.patch.object(gate, 'current_provenance', side_effect=[provenance(), provenance(), changed]):
            with self.assertRaisesRegex(ValueError, 'bindings changed before acceptance return'):
                self.collect()
        with mock.patch.object(gate, 'review_sources', side_effect=[self.sources, self.sources, {'changed': True}]):
            with self.assertRaisesRegex(ValueError, 'bindings changed before acceptance return'):
                self.collect()

    def test_exact_audit_witness_bytes_are_rechecked_after_collection(self):
        validate = gate.validate_pairs

        def mutate_audit_after_capture(records, seeds):
            result = validate(records, seeds)
            audit = historical_gate.CENTRAL_AUDIT
            audit.write_text(' ' + audit.read_text())
            return result

        with mock.patch.object(gate, 'validate_pairs', side_effect=mutate_audit_after_capture):
            with self.assertRaises(ValueError):
                self.collect()

    def test_exact_pair_coverage_required(self):
        records = {(scenario, seed): self.result(scenario, seed)
                   for scenario in protocol.SCENARIOS for seed in protocol.PILOT_SEEDS}
        records.pop(('iid', 0))
        with self.assertRaisesRegex(ValueError, 'incomplete MLP8 paired coverage'):
            gate.validate_pairs(records, list(protocol.PILOT_SEEDS))

    def test_source_drift_inventory_path_symlink_and_noncanonical_proof_rejected(self):
        with mock.patch.object(gate, 'current_provenance', side_effect=[
                provenance(), dict(provenance(), source_tree_sha256='e' * 64)]):
            with self.assertRaisesRegex(ValueError, 'changed during acceptance'):
                self.collect()
        alias = gate.INVENTORY_ROOT / 'symlink.json'
        alias.symlink_to(self.inventory_path)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            gate.collect_acceptance(alias)
        path = gate.PROOF_ROOT / 'noncanonical.json'
        write_json(path, self.collect())
        with self.assertRaisesRegex(ValueError, 'invalid same-source MLP8 pilot proof'):
            gate.verify_pilot_acceptance(path)

    def test_unexplained_producer_stderr_rejected(self):
        job_id = self.inventory['requests'][0]['job_id']
        path = historical_gate.CENTRAL_LOGS / (job_id + '.attempt-001.stderr.log')
        path.write_text('Traceback: unexplained scientific failure\n')
        with self.assertRaises(ValueError):
            self.collect()

    def test_formal_cannot_use_missing_old_or_wrong_source_pilot(self):
        specs = build_requests(cohort='formal')
        records = [dict(parameters=spec['request']['parameters']) for spec in specs]
        inventory = self.inventory_for('formal', specs, records)
        path = gate.INVENTORY_ROOT / 'formal.inventory.json'
        write_json(path, inventory)
        with self.assertRaises((ValueError, FileNotFoundError)):
            gate.collect_acceptance(path, 'formal')
        proof_path = gate.default_acceptance_path()
        for scope, source in (('survival_pooled_pilot_scientific_acceptance', provenance()),
                              ('survival_mlp8_pilot_scientific_acceptance', dict(provenance(), git_commit='f' * 40))):
            write_json(proof_path, dict(scope=scope, passed=True, provenance=source))
            with self.assertRaisesRegex(ValueError, 'invalid same-source MLP8 pilot proof'):
                gate.collect_acceptance(path, 'formal')

    def test_formal_receives_fresh_current_mlp8_proof_digest(self):
        proof = self.collect()
        proof_path = gate.publish_acceptance(proof)
        specs = build_requests(cohort='formal')
        records = [self.request_record(spec, gate.OUTBOX_ROOT / 'formal' / spec['filename']) for spec in specs]
        write_json(Path(records[0]['path']), specs[0]['request'])
        self.write_completed_job(specs[0]['request'])
        inventory = self.inventory_for('formal', specs, records)
        inventory['pilot_acceptance'] = dict(path=str(proof_path), sha256=checksum(proof_path), evidence=proof)
        path = gate.INVENTORY_ROOT / 'formal.inventory.json'
        write_json(path, inventory)
        factory, seen = gate.ArtifactStore, []

        def stop_after_admission(config, source, prerequisite=None):
            if config['task'] == protocol.FORMAL_TASK:
                seen.append(prerequisite)
                raise RuntimeError('formal admission reached; no formal inference or fitting')
            return factory(config, source, prerequisite)

        with mock.patch.object(gate, 'ArtifactStore', side_effect=stop_after_admission):
            with self.assertRaisesRegex(RuntimeError, 'formal admission reached'):
                gate.collect_acceptance(path, 'formal')
        self.assertEqual(seen, [dict(pilot_acceptance_sha256=gate.digest(proof),
                                    provenance_sha256=provenance()['provenance_sha256'])])
        self.assertEqual(len(self.status_calls), 8)

    def test_immutable_proof_publication_and_existing_partial_preserved(self):
        proof = self.collect()
        output = gate.publish_acceptance(proof)
        original = output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'existing or partial'):
            gate.publish_acceptance(dict(proof, validated_unit_count=999))
        self.assertEqual(output.read_bytes(), original)
        output.unlink()  # Test-owned fixture only; actual proofs remain immutable.
        pending = output.with_name('.' + output.name + '.pending')
        pending.write_bytes(b'preserve original partial fixture')
        with self.assertRaisesRegex(ValueError, 'existing or partial'):
            gate.publish_acceptance(proof)
        self.assertEqual(pending.read_bytes(), b'preserve original partial fixture')
        self.assertFalse(output.exists())

    def test_proof_root_symlink_and_scope_rejected_before_write(self):
        proof = self.collect()
        destination = self.root / 'other-proof-root'
        destination.mkdir()
        gate.PROOF_ROOT.symlink_to(destination, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            gate.publish_acceptance(proof)
        self.assertEqual(list(destination.iterdir()), [])
        gate.PROOF_ROOT.unlink()
        for changes in (dict(scope='survival_pooled_pilot_scientific_acceptance'),
                        dict(provenance=dict(provenance(), git_dirty='1')),
                        dict(passed=False)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'publication scope/source mismatch'):
                gate.publish_acceptance(dict(proof, **changes))

    def test_source_drift_before_and_after_staging_is_fail_closed(self):
        proof = self.collect()
        output = gate.default_acceptance_path()
        changed = dict(provenance(), source_tree_sha256='e' * 64)
        with mock.patch.object(gate, 'current_provenance', side_effect=[provenance(), changed]):
            with self.assertRaisesRegex(ValueError, 'before MLP8 proof publication'):
                gate.publish_acceptance(proof)
        pending = output.with_name('.' + output.name + '.pending')
        self.assertFalse(output.exists())
        self.assertFalse(pending.exists())
        with mock.patch.object(gate, 'current_provenance', side_effect=[provenance(), provenance(), changed]):
            with self.assertRaisesRegex(ValueError, 'preserve partial MLP8 proof'):
                gate.publish_acceptance(proof)
        self.assertFalse(output.exists())
        self.assertTrue(pending.exists())
        self.assertEqual(json.loads(pending.read_text()), proof)


if __name__ == '__main__':
    unittest.main()
