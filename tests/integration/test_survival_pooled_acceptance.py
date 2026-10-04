"""Fabricated saved-evidence acceptance: no fitting, inference or central writes.

The store boundary has separate numerical tests. Here it is replaced by a tiny
hash-checking store while request bytes, completed jobs, producing attempts,
retries, exact audit witnesses, paired coverage and fresh proof replay stay real.
Every fixture lives below the FedFisher scratch root.
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
from effect_modifier_experiment.survival_pooled import acceptance as gate, protocol
from effect_modifier_experiment.survival_pooled.requests import build_requests


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


class SurvivalPooledAcceptanceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-pooled-validation/acceptance')
        gate._ordinary_path(scratch, exists=False)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        locations = dict(OUTBOX_ROOT=self.root / 'outbox', INVENTORY_ROOT=self.root / 'reports',
                         PROOF_ROOT=self.root / 'proofs', CENTRAL_JOBS=self.root / 'central/jobs')
        for key, value in locations.items():
            self.stack.enter_context(mock.patch.object(gate, key, value))
        # Imported producer/log helpers retain their defining module's globals.
        # Patch those roots to fabricated local evidence rather than mocking the
        # meaningful ownership, receipt, retry and audit-validation mechanisms.
        for key, suffix in (('CENTRAL_LOGS', 'central/logs'),
                            ('CENTRAL_RETRIES', 'central/retries'),
                            ('CENTRAL_AUDIT', 'central/audit.jsonl')):
            self.stack.enter_context(mock.patch.object(historical_gate, key, self.root / suffix))
            if key in ('CENTRAL_LOGS', 'CENTRAL_RETRIES'):
                self.stack.enter_context(mock.patch.object(gate, key, self.root / suffix))
        self.current = self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
        self.historical = {'fixture_acceptance': {'path': str(self.root / 'old.json'), 'sha256': '1' * 64}}
        self.stack.enter_context(mock.patch.object(gate, 'historical_binding', return_value=self.historical))
        self.validate_historical = self.stack.enter_context(mock.patch.object(gate, 'validate_historical_inputs'))
        self.sources = {'fixture': {'path': str(self.root / 'source.py'), 'sha256': '2' * 64}}
        self.reviewed = self.stack.enter_context(mock.patch.object(gate, 'review_sources', return_value=self.sources))
        self.directories, self.status_calls, self.prerequisites = {}, [], []
        self.specs = build_requests()
        requests, audit = [], []
        for spec in self.specs:
            raw, config = spec['request'], spec['configuration']
            request_path = gate.OUTBOX_ROOT / 'bundle' / spec['filename']
            write_json(request_path, raw)
            requests.append(dict(path=str(request_path), sha256=checksum(request_path),
                job_id=raw['job_id'], task=raw['task'], parameters=raw['parameters'],
                seeds=spec['seed_sequence'], run_id=spec['run_id'], configuration=config,
                configuration_sha256=spec['configuration_sha256']))
            p = raw['parameters']
            parent = self.root / 'outputs' / config['run_id']
            write_json(parent / 'run_manifest.env', {'fixture_config': config})
            for seed in spec['seed_sequence']:
                directory = parent / ('seed_%03d' % seed)
                directory.mkdir()
                self.directories[p['family'], p['scenario'], seed] = directory
                write_json(directory / 'started.json', dict(execution=execution(raw['job_id'])))
                write_json(directory / 'result.json', self.result(p['family'], p['scenario'], seed))
                (directory / 'model.npz').write_bytes(b'fabricated model boundary; no numerical inference')
                self.rehash(directory)
            job = dict(raw, state='completed', exit_code=0, **{key: value for key, value
                       in execution(raw['job_id']).items() if key not in ('job_id', 'attempt')})
            for kind in ('stdout', 'stderr'):
                path = historical_gate.CENTRAL_LOGS / ('%s.attempt-001.%s.log' % (raw['job_id'], kind))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('')
                job[kind + '_log'] = str(path)
            write_json(gate.CENTRAL_JOBS / (raw['job_id'] + '.json'), job)
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
                self.config = config
                self.directory = owner.root / 'outputs' / config['run_id']
                owner.prerequisites.append(prerequisite)

            def cell_dir(self, seed):
                return self.directory / ('seed_%03d' % seed)

            def status(self, seed):
                owner.status_calls.append((self.config['family'], self.config['scenario'], seed))
                directory = self.cell_dir(seed)
                marker = json.loads((directory / self.done_name).read_text())
                if any(checksum(directory / name) != value for name, value in marker.items()):
                    raise ValueError('fabricated artifact checksum mismatch')
                return 'validated'

        self.stack.enter_context(mock.patch.object(gate, 'ArtifactStore', FabricatedStore))

    def inventory_for(self, cohort, specs, requests):
        seeds = list(protocol.PILOT_SEEDS if cohort == 'pilot' else protocol.FORMAL_SEEDS)
        units = 6 * len(seeds)
        return dict(schema_version=1, bundle='fedfisher-survival-pooled-v1-' + cohort,
            generation=1, task=protocol.PILOT_TASK if cohort == 'pilot' else protocol.FORMAL_TASK,
            request_count=len(specs), scientific_cell_count=units, pooled_fit_count=2 * units,
            coverage=dict(families=list(protocol.FAMILIES), scenarios=list(protocol.SCENARIOS),
                          seeds=seeds, pseudo_scopes=list(protocol.PSEUDO_SCOPES)),
            provenance=provenance(), project_head=provenance()['git_commit'], worktree_status='clean',
            bundle_ready=True, review_sources=self.sources, historical_inputs=self.historical, requests=requests)

    def result(self, family, scenario, seed):
        # Deliberately poor accuracy remains a valid scientific observation.
        summary = dict(arms={'0': {'rmse': .9}, '1': {'rmse': .8}}, combined={'rmse': .85},
                       factual={'rmse': .9}, contrast={'rmse': 10.})
        tags = protocol.historical_tags(family)
        return dict(status='complete', site_sizes=list(range(500, 510)),
            fingerprints=dict(training=hashlib.sha256((scenario + str(seed)).encode()).hexdigest(),
                              evaluation=hashlib.sha256(('x' + str(seed)).encode()).hexdigest(),
                              evaluation_truth=hashlib.sha256(('truth' + str(seed)).encode()).hexdigest()),
            prediction={tag: deepcopy(summary) for tag in (*tags, *('pooled_' + s for s in protocol.PSEUDO_SCOPES))},
            comparisons={tag: {scope: {'combined': {'rmse': .8}} for scope in protocol.PSEUDO_SCOPES}
                         for tag in tags},
            fit={scope: dict(family=family, pseudo_scope=scope, sample_count=5045,
                            total_fit_seconds=1., stop_reason='stationary' if family == 'linear' else 'fixed_epoch_budget')
                 for scope in protocol.PSEUDO_SCOPES},
            timing=dict(stages={name: {'wall_seconds': 3., 'cpu_seconds': 2.}
                                for name in ('historical_input_validation', 'data_preparation', 'fit', 'probability_evaluation')}),
            memory={'process_peak_rss_kib': 1000, 'scope': 'fabricated test observation'})

    def rehash(self, directory):
        write_json(directory / 'done.json', {name: checksum(directory / name)
                   for name in ('started.json', 'model.npz', 'result.json')})

    def change_result(self, key, mutate):
        directory = self.directories[key]
        result = json.loads((directory / 'result.json').read_text())
        mutate(result)
        write_json(directory / 'result.json', result)
        self.rehash(directory)

    def test_exact_twelve_saved_units_and_fresh_proof_replay_without_inference(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        self.assertEqual(proof['validated_request_count'], 6)
        self.assertEqual(proof['validated_unit_count'], 12)
        self.assertEqual(len(self.status_calls), 12)
        self.assertEqual(proof['paired_checks']['complete_seed_count'], 2)
        self.assertTrue(proof['paired_checks']['passed'])
        self.assertFalse(proof['inference_performed'])
        self.assertTrue(all(p is None for p in self.prerequisites))
        self.assertTrue(all(not u['observations']['performance_threshold_applied'] for u in proof['units']))
        self.assertEqual(proof['units'][0]['observations']['probability_accuracy']['dslasso']['contrast']['rmse'], 10.)
        self.assertGreaterEqual(self.validate_historical.call_count, 2)
        write_json(gate.default_acceptance_path(), proof)
        self.assertEqual(gate.verify_pilot_acceptance(gate.default_acceptance_path()), proof)
        self.assertEqual(len(self.status_calls), 24)

    def test_audit_scans_once_per_producing_attempt_while_rechecking_every_unit(self):
        with mock.patch.object(gate, '_attempt_audit', wraps=gate._attempt_audit) as scans, \
             mock.patch.object(gate, '_recheck_producer', wraps=gate._recheck_producer) as rechecks:
            proof = gate.collect_pilot_acceptance(self.inventory_path)
        self.assertEqual(proof['validated_unit_count'], 12)
        self.assertEqual(scans.call_count, 6)
        expected = {(row['job_id'], 1) for row in self.inventory['requests']}
        self.assertEqual({call.args for call in scans.call_args_list}, expected)
        self.assertEqual(rechecks.call_count, 12)

    def test_saved_passed_boolean_or_changed_summary_cannot_open_formal_gate(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        proof['validated_unit_count'] = 999
        write_json(gate.default_acceptance_path(), proof)
        with self.assertRaisesRegex(ValueError, 'freshly verified'):
            gate.verify_pilot_acceptance(gate.default_acceptance_path())

    def test_request_bytes_inventory_counts_and_duplicate_recipe_fail_closed(self):
        original = deepcopy(self.inventory)
        changed = deepcopy(original)
        changed['requests'][-1] = changed['requests'][0]
        write_json(self.inventory_path, changed)
        with self.assertRaisesRegex(ValueError, 'request differs'):
            gate.collect_pilot_acceptance(self.inventory_path)
        changed = deepcopy(original)
        changed['request_count'] = 5
        write_json(self.inventory_path, changed)
        with self.assertRaisesRegex(ValueError, 'request inventory'):
            gate.collect_pilot_acceptance(self.inventory_path)
        write_json(self.inventory_path, original)
        Path(original['requests'][0]['path']).write_bytes(b'{}\n')
        with self.assertRaisesRegex(ValueError, 'request differs'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_source_coverage_and_historical_anchor_changes_are_rejected(self):
        for changes in (dict(provenance=dict(provenance(), source_tree_sha256='e' * 64)),
                        dict(coverage=dict(self.inventory['coverage'], seeds=[0])),
                        dict(historical_inputs={'different': True}), dict(pooled_fit_count=12)):
            write_json(self.inventory_path, dict(self.inventory, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'frozen source'):
                gate.collect_pilot_acceptance(self.inventory_path)
        write_json(self.inventory_path, self.inventory)
        with mock.patch.object(gate, 'review_sources', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'reviewed pooled sources'):
                gate.collect_pilot_acceptance(self.inventory_path)

    def test_wrong_job_owner_state_parameters_and_boolean_exit_zero_rejected(self):
        request = self.inventory['requests'][0]
        path = gate.CENTRAL_JOBS / (request['job_id'] + '.json')
        original = json.loads(path.read_text())
        for changes in (dict(state='failed'), dict(project='Other'), dict(exit_code=False),
                        dict(task=protocol.FORMAL_TASK), dict(parameters={'family': 'wrong'})):
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'matching completed'):
                gate.collect_pilot_acceptance(self.inventory_path)

    def test_actual_producer_must_match_job_and_attempt_audit(self):
        key = ('linear', 'iid', 0)
        directory = self.directories[key]
        started = json.loads((directory / 'started.json').read_text())
        started['execution']['job_id'] = 'fedfisher-survival-foreign-producer'
        write_json(directory / 'started.json', started)
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError, 'another scheduler job'):
            gate.collect_pilot_acceptance(self.inventory_path)
        started['execution']['job_id'] = self.inventory['requests'][0]['job_id']
        write_json(directory / 'started.json', started)
        self.rehash(directory)
        audit_path = historical_gate.CENTRAL_AUDIT
        lines = [line for line in audit_path.read_text().splitlines()
                 if self.inventory['requests'][0]['job_id'] not in line]
        audit_path.write_text('\n'.join(lines) + '\n')
        with self.assertRaisesRegex(ValueError, 'exact central start/running audit'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_retry_preserves_earlier_producer_and_observed_fit_time(self):
        request = self.inventory['requests'][0]
        job_id = request['job_id']
        job_path = gate.CENTRAL_JOBS / (job_id + '.json')
        job = json.loads(job_path.read_text())
        job['estimated_runtime_seconds'] = 90.
        for kind in ('stdout', 'stderr'):
            log = historical_gate.CENTRAL_LOGS / ('%s.attempt-002.%s.log' % (job_id, kind))
            log.write_text('')
            job[kind + '_log'] = str(log)
        write_json(job_path, job)
        retry_path = historical_gate.CENTRAL_RETRIES / (job_id + '.json')
        retry = json.loads(retry_path.read_text())
        retry['attempts_started'] = 2
        write_json(retry_path, retry)
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        producer = proof['units'][0]['producer_evidence']
        self.assertEqual((producer['producer_attempt'], producer['final_completed_attempt']), (1, 2))
        self.assertTrue(producer['resumed_from_earlier_attempt'])
        self.assertEqual(proof['units'][0]['measured_stage_wall_seconds'], 12.)

    def test_saved_artifact_hash_failure_and_paired_data_mismatch_fail(self):
        directory = self.directories['deep', 'iid', 0]
        (directory / 'model.npz').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            gate.collect_pilot_acceptance(self.inventory_path)
        self.rehash(directory)
        self.change_result(('deep', 'iid', 0), lambda result:
                           result['fingerprints'].update(training='e' * 64))
        with self.assertRaisesRegex(ValueError, 'different training/test/truth'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_cross_scenario_common_test_and_exact_pair_coverage_are_required(self):
        records = {(f, s, seed): self.result(f, s, seed) for f in protocol.FAMILIES
                   for s in protocol.SCENARIOS for seed in protocol.PILOT_SEEDS}
        for family in protocol.FAMILIES:
            records[family, 'covariate_shift', 49]['fingerprints']['evaluation_truth'] = 'e' * 64
        with self.assertRaisesRegex(ValueError, 'different independent test'):
            gate.validate_pairs(records, list(protocol.PILOT_SEEDS))
        records.pop(('deep', 'iid', 0))
        with self.assertRaisesRegex(ValueError, 'incomplete pooled survival paired coverage'):
            gate.validate_pairs(records, list(protocol.PILOT_SEEDS))

    def test_source_drift_during_acceptance_and_noncanonical_proof_are_rejected(self):
        with mock.patch.object(gate, 'current_provenance', side_effect=[
                provenance(), dict(provenance(), source_tree_sha256='e' * 64)]):
            with self.assertRaisesRegex(ValueError, 'changed during pooled acceptance'):
                gate.collect_pilot_acceptance(self.inventory_path)
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        other = gate.PROOF_ROOT / 'noncanonical.json'
        write_json(other, proof)
        with self.assertRaisesRegex(ValueError, 'canonical for current clean source'):
            gate.verify_pilot_acceptance(other)

    def test_formal_inventory_cannot_reuse_missing_or_wrong_source_pilot_proof(self):
        specs = build_requests(cohort='formal')
        requests = [dict(parameters=spec['request']['parameters']) for spec in specs]
        inventory = self.inventory_for('formal', specs, requests)
        path = gate.INVENTORY_ROOT / 'formal.inventory.json'
        write_json(path, inventory)
        with self.assertRaises((ValueError, FileNotFoundError)):
            gate.collect_acceptance(path, 'formal')
        proof = dict(scope='survival_pooled_pilot_scientific_acceptance', passed=True,
                     provenance=dict(provenance(), git_commit='f' * 40))
        write_json(gate.default_acceptance_path(), proof)
        with self.assertRaisesRegex(ValueError, 'invalid pooled pilot scientific proof'):
            gate.collect_acceptance(path, 'formal')

    def test_formal_store_receives_digest_of_freshly_verified_canonical_pilot(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        proof_path = gate.default_acceptance_path()
        write_json(proof_path, proof)
        specs = build_requests(cohort='formal')
        records = []
        for spec in specs:
            raw = spec['request']
            path = gate.OUTBOX_ROOT / 'formal-fixture' / spec['filename']
            records.append(dict(path=str(path), sha256=spec['request_sha256'],
                job_id=raw['job_id'], task=raw['task'], parameters=raw['parameters'],
                seeds=spec['seed_sequence'], run_id=spec['run_id'], configuration=spec['configuration'],
                configuration_sha256=spec['configuration_sha256']))
        # The test stops at the first formal store constructor. No formal
        # result, dataset, model or 1,800-unit output fixture is needed.
        raw = specs[0]['request']
        write_json(Path(records[0]['path']), raw)
        job = dict(raw, state='completed', exit_code=0, **{key: value for key, value
                   in execution(raw['job_id']).items() if key not in ('job_id', 'attempt')})
        for kind in ('stdout', 'stderr'):
            path = historical_gate.CENTRAL_LOGS / ('%s.attempt-001.%s.log' % (raw['job_id'], kind))
            path.write_text('')
            job[kind + '_log'] = str(path)
        write_json(gate.CENTRAL_JOBS / (raw['job_id'] + '.json'), job)
        inventory = self.inventory_for('formal', specs, records)
        inventory['pilot_acceptance'] = dict(path=str(proof_path), sha256=checksum(proof_path), evidence=proof)
        inventory_path = gate.INVENTORY_ROOT / 'formal.inventory.json'
        write_json(inventory_path, inventory)
        factory, seen = gate.ArtifactStore, []

        def stop_after_admission(config, source, prerequisite=None):
            if config['task'] == protocol.FORMAL_TASK:
                seen.append(prerequisite)
                raise RuntimeError('formal admission reached; fixture stops before scientific outputs')
            return factory(config, source, prerequisite=prerequisite)

        with mock.patch.object(gate, 'ArtifactStore', side_effect=stop_after_admission):
            with self.assertRaisesRegex(RuntimeError, 'formal admission reached'):
                gate.collect_acceptance(inventory_path, 'formal')
        self.assertEqual(seen, [dict(pilot_acceptance_sha256=gate.digest(proof),
                                   provenance_sha256=provenance()['provenance_sha256'])])
        self.assertEqual(len(self.status_calls), 24)  # Actual fresh pilot replay.


if __name__ == '__main__':
    unittest.main()
