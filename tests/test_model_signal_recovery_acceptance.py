"""Adversarial saved-evidence fixtures; no training, inference or generated datasets."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.model_signal_ablation import recovery_acceptance as gate
from effect_modifier_experiment.model_signal_ablation import recovery_protocol as protocol
from effect_modifier_experiment.model_signal_ablation import recovery
from scripts import prepare_model_signal_recovery_requests as builder


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance():
    return dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                source_tree_sha256='c' * 64, provenance_sha256='d' * 64)


def _receipt(job_id, attempt=1):
    return dict(job_id=job_id, attempt=attempt, execution_profile='cpu-1core', estimated_runtime_seconds=20.,
        allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0, gpu_memory_mib=0,
                        gpu_utilization_pct=0, exclusive_gpu=False))


def _result(key):
    setting, cell, variant, seed, split = key
    fingerprint = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    data = (setting, seed, split, 'shift' if 'site_gaussian' in variant else 'control')
    local = 'local_permutation' in variant
    method = protocol.CELLS[cell]
    fit = dict(fit_fingerprint=fingerprint((data, cell)), numeric_parameter=.25,
        round_history=[dict(gradient_updates=4, training_wall_seconds=10., training_cpu_seconds=9.)],
        timing=dict(fit_wall_seconds=20.))
    if method['family'] == 'legacy_fedfisher':
        fit.update(initial_model_fingerprint=fingerprint(method['model']),
            local_model_fingerprints=[fingerprint((data, method['model']))], validation=dict(data=str(data)))
    return dict(status='complete', site_sizes=[500, 500], label_counts={'0': 500, '1': 500},
        fingerprints=dict(training=fingerprint(data), test=fingerprint((setting, seed, split)),
                          model=fingerprint((data, cell))), fit=fit,
        evaluation=dict(raw_x_sha256=fingerprint((setting, seed, split, 'x')),
            permutation_sha256=fingerprint((setting, seed, split, local)),
            shard_indices_sha256=[fingerprint(local)], shard_sizes=[10000], metrics=dict(score=.5)),
        timing=dict(wall_seconds=20.), memory=dict(process_peak_rss_kib=1000))


def _observations():
    baseline, diagnostic, controls, caps = gate._pilot_sets()
    records = {key: dict(outcome='noncompleted' if key in caps else 'validated', result=_result(key))
               for key in baseline | diagnostic}
    references = {key: deepcopy(records[key]['result']) for key in baseline | controls}
    return records, references


class RecoveryParityTests(unittest.TestCase):
    def test_exact_582_parities_two_caps_and_four_later_terminals(self):
        records, references = _observations()
        proof = gate.validate_pilot_observations(records, references)
        self.assertEqual((proof['baseline_success_parity'], proof['diagnostic_success_parity'],
                          proof['expected_caps_reproduced'], proof['seed41_terminal_units']), (576, 6, 2, 4))
        for key, record in records.items():
            if key[3] == 41:
                record['outcome'] = 'noncompleted'
        self.assertTrue(gate.validate_pilot_observations(records, references)['passed'])

    def test_control_must_stay_success_and_known_cap_must_recur(self):
        _, _, controls, caps = gate._pilot_sets()
        cases = [(next(iter(controls)), 'noncompleted'), (next(iter(caps)), 'validated')]
        for key, outcome in cases:
            records, references = _observations()
            records[key]['outcome'] = outcome
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.validate_pilot_observations(records, references)
        records, references = _observations()
        records.pop(next(key for key in records if key[3] == 41))
        with self.assertRaisesRegex(ValueError, '588'):
            gate.validate_pilot_observations(records, references)

    def test_parity_ignores_only_measurements_and_rejects_any_scientific_change(self):
        key = ('small_mlp', 'fedavg_small_e1_patience', 'control', 0, 'iid')
        reference = _result(key)
        changed = deepcopy(reference)
        changed['memory']['process_peak_rss_kib'] *= 2
        changed['fit']['timing']['fit_wall_seconds'] *= 2
        changed['fit']['round_history'][0]['training_wall_seconds'] *= 3
        changed['identity'] = dict(run_id='new-epoch')
        self.assertTrue(gate.validate_numeric_parity(changed, reference))
        changes = [lambda r: r['fit'].update(numeric_parameter=.26),
                   lambda r: r['fit']['round_history'][0].update(gradient_updates=5),
                   lambda r: r['fingerprints'].update(model='x' * 64),
                   lambda r: r['evaluation']['metrics'].update(score=.51)]
        for change in changes:
            record = deepcopy(changed)
            change(record)
            with self.assertRaisesRegex(ValueError, 'numeric fitting path'):
                gate.validate_numeric_parity(record, reference)

    def test_scientific_noncompletion_is_included_in_terminal_training_pairing(self):
        left = ('small_mlp', 'fedavg_small_e1_patience', 'site_gaussian_shift', 25, 'noniid')
        right = (*left[:2], 'site_gaussian_local_permutation', *left[3:])
        records = {key: gate._terminal_thin(_result(key), 'noncompleted') for key in (left, right)}
        self.assertEqual(gate.validate_terminal_training_pairs(records)['terminal_training_pairs'], 1)
        for key, value in (('outcome', 'validated'), ('numeric_fit_sha256', '0' * 64)):
            changed = deepcopy(records)
            changed[right][key] = value
            with self.assertRaisesRegex(ValueError, 'completion state'):
                gate.validate_terminal_training_pairs(changed)
        with self.assertRaisesRegex(ValueError, 'planned peer'):
            gate.validate_terminal_training_pairs({left: records[left]})


class RecoveryCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retained, cls.replacement = [], []
        for setting, recipe in protocol.SETTINGS.items():
            for cell in recipe['cells']:
                for variant in protocol.VARIANTS:
                    for seed in range(300):
                        for split in protocol.SPLITS:
                            row = dict(setting=setting, cell=cell, variant=variant, seed=seed, split=split)
                            if len(cls.retained) < 4838:
                                cls.retained.append(dict(row, classification=gate.ACCEPTED_OLD, producer_key='real-old-producer'))
                            else:
                                cls.replacement.append(dict(row, outcome='validated'))

    def test_exact_mixed_source_coverage_retains_all_300_seed_denominators(self):
        records = list(self.replacement)
        index = next(i for i, row in enumerate(records) if row['cell'].endswith('_patience'))
        records[index] = dict(records[index], outcome='noncompleted')
        proof = gate.validate_composite_coverage(self.retained, records)
        self.assertEqual((proof['scientific_units'], proof['completed_units'], proof['noncompleted_units']),
                         (172800, 172799, 1))
        self.assertFalse(proof['whole_cohort_scientific_success'])
        self.assertTrue(all(row['planned_seeds'] == 300 for row in proof['conditions']))
        affected = next(row for row in proof['conditions'] if row['noncompleted'])
        self.assertEqual(affected['completion_rate'], 299/300)

    def test_duplicate_missing_old_partial_and_unreviewed_failure_cannot_fill_coverage(self):
        with self.assertRaisesRegex(ValueError, '167962'):
            gate.validate_composite_coverage(self.retained, self.replacement[:-1])
        changed = list(self.replacement)
        changed[-1] = changed[-2]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            gate.validate_composite_coverage(self.retained, changed)
        historical = list(self.retained)
        historical[0] = dict(historical[0], classification='artifact_valid_with_unknown_central_exit')
        with self.assertRaisesRegex(ValueError, 'excluded old'):
            gate.validate_composite_coverage(historical, self.replacement)
        changed = list(self.replacement)
        index = next(i for i, row in enumerate(changed) if row['cell'] == 'dslassocov')
        changed[index] = dict(changed[index], outcome='noncompleted')
        with self.assertRaisesRegex(ValueError, 'beyond patience'):
            gate.validate_composite_coverage(self.retained, changed)


class RecoveryRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts import prepare_model_signal_requests as old_builder
        rows = []
        for spec in old_builder.build_requests(1, 'formal', 16):
            request = spec['request']
            rows.append(dict(job_id=request['job_id'], parameters=request['parameters'], task=request['task'],
                configuration=spec['configuration'], configuration_sha256=spec['configuration_sha256'],
                run_id=spec['run_id'], seeds=spec['seed_sequence'], splits=spec['splits']))
        failed = {row['job_id'] for row in rows if row['parameters']['setting'] == 'small_mlp'
            and row['parameters']['cell'] == 'fedavg_small_e1_patience'
            and row['parameters']['variant'] in protocol.DIAGNOSTIC_VARIANTS
            and row['parameters']['shard_index'] == 9}
        retained, totals = set(), {19: 0, 18: 0}
        for row in rows:
            count = len(row['seeds'])
            if row['job_id'] not in failed and totals[count] < {19: 97, 18: 32}[count]:
                retained.add(row['job_id'])
                totals[count] += 1
        unknown = next(row['job_id'] for row in rows
            if row['job_id'] not in retained | failed and len(row['seeds']) == 19)
        units, snapshots, producers = [], [], {}
        for row in rows:
            jid, p = row['job_id'], row['parameters']
            snapshots.append(dict(job_id=jid, parameters=p, seeds=row['seeds'],
                planned_unit_count=2*len(row['seeds']), state='completed' if jid in retained else 'failed'
                    if jid in failed | {unknown} else 'not_admitted'))
            if jid in retained:
                producers[jid] = dict(execution_receipt=dict(job_id=jid))
            for seed in row['seeds']:
                for split in protocol.SPLITS:
                    if jid not in retained | {unknown} and not (jid in failed and
                            (seed, split) in ((9, 'iid'), (9, 'noniid'), (25, 'iid'))):
                        continue
                    units.append(dict(setting=p['setting'], cell=p['cell'], variant=p['variant'],
                        seed=seed, split=split, job_id=jid, producer_key=jid if jid in retained else None,
                        classification=gate.ACCEPTED_OLD if jid in retained else
                            'artifact_valid_in_failed_scientific_shard' if jid in failed else
                            'artifact_valid_with_unknown_central_exit'))
        binding = dict(path='/scr/del6500/FedFisher/fabricated-old-inventory.json', sha256='e'*64)
        cls.inventory = dict(provenance=_provenance(), task=gate.original.FORMAL_TASK,
            request_count=4608, scientific_cell_count=172800, requests=rows)
        cls.formal = dict(source_provenance=_provenance(), existing_saved_artifact_checks_passed=True,
            whole_cohort_accepted=False, no_active_cohort_jobs=True, accepted_successful_shard_count=129,
            accepted_successful_unit_count=4838, formal_inventory=binding, requests=snapshots,
            units=units, producer_evidence=producers, failures=[dict(job_id=jid) for jid in sorted(failed)])
        cls.pilot = dict(provenance=_provenance(), passed=True, validated_request_count=288, validated_unit_count=576)
        cls.proof = dict(counts=gate.COUNTS, provenance=_provenance(), formal_inventory=binding,
            retained_job_ids=sorted(retained), replacement_job_ids=sorted({r['job_id'] for r in rows}-retained),
            known_success_diagnostic_controls=[u for u in units if u['classification'] == 'artifact_valid_in_failed_scientific_shard'],
            known_exhaustion_controls=cls.formal['failures'])

    def test_exact_retention_excludes_all_44_and_preserves_six_as_diagnostic_only(self):
        accepted, excluded = gate._validate_retention(self.proof, self.formal, self.inventory, self.pilot)
        self.assertEqual((len(accepted), len(excluded)), (4838, 44))
        self.assertEqual(len(self.proof['known_success_diagnostic_controls']), 6)

    def test_partial_promotions_changed_producers_and_missing_old_units_fail(self):
        changed = dict(self.formal, units=list(self.formal['units']))
        index = next(i for i, row in enumerate(changed['units']) if row['classification'] != gate.ACCEPTED_OLD)
        changed['units'][index] = dict(changed['units'][index], classification=gate.ACCEPTED_OLD)
        with self.assertRaisesRegex(ValueError, 'centrally successful producer'):
            gate._validate_retention(self.proof, changed, self.inventory, self.pilot)
        changed = dict(self.formal, producer_evidence=dict(self.formal['producer_evidence']))
        key = next(iter(changed['producer_evidence']))
        changed['producer_evidence'][key] = dict(execution_receipt=dict(job_id='foreign'))
        with self.assertRaisesRegex(ValueError, 'producer identity'):
            gate._validate_retention(self.proof, changed, self.inventory, self.pilot)
        with self.assertRaisesRegex(ValueError, 'coverage changed'):
            gate._validate_retention(self.proof, dict(self.formal, units=self.formal['units'][1:]), self.inventory, self.pilot)


class RecoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        root = Path('/scr/del6500/FedFisher/model-signal-validation/recovery-acceptance')
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_historical_source_binding_is_distinct_from_current_review_source(self):
        source = self.root / 'source' / 'observer.py'
        source.parent.mkdir()
        source.write_text('old bytes')
        original = gate._binding(source)
        source.write_text('new observer bytes')
        with mock.patch.object(gate, 'PROJECT_ROOT', source.parent):
            gate._EvidenceSnapshot().add(gate._binding(source))
            gate._EvidenceSnapshot({str(source): original}).add(original)
            with self.assertRaisesRegex(ValueError, 'frozen source inventory'):
                gate._EvidenceSnapshot({str(source): original}).add(gate._binding(source))
            with self.assertRaisesRegex(ValueError, 'frozen source inventory'):
                gate._EvidenceSnapshot({}).add(original)

    def test_persistent_reference_hash_drift_and_late_file_drift_are_rejected(self):
        path = self.root / 'old-result.json'
        _write(path, {'verified_science': True})
        bound = gate._binding(path)
        snapshot = gate._EvidenceSnapshot()
        snapshot.add(bound)
        _write(path, {'verified_science': False})
        with self.assertRaisesRegex(ValueError, 'evidence bytes changed'):
            snapshot.recheck()
        with self.assertRaisesRegex(ValueError, 'bound recovery evidence changed'):
            gate._checked(bound)

    def test_fixed_closeout_path_and_digest_cannot_be_replaced_by_a_passed_boolean(self):
        path = self.root / 'forged-closeout.json'
        _write(path, dict(source_release_passed=True, whole_cohort_accepted=False))
        with self.assertRaisesRegex(ValueError, 'fixed reviewed'):
            gate.verify_closeout(path)
        with mock.patch.object(protocol, 'SOURCE_CLOSEOUT', dict(path=str(path), sha256='0' * 64)):
            with self.assertRaisesRegex(ValueError, 'bound recovery evidence changed'):
                gate.verify_closeout(path)

    def test_audit_batch_matches_original_range_contract_and_detects_duplicate_attempt(self):
        path = self.root / 'events.jsonl'
        jid = 'fedfisher-ms-fixture'
        events = [dict(schema_version=2, job_id=jid, event='job_attempt_started', attempt=1),
            dict(schema_version=2, job_id=jid, event='job_transition', project='FedFisher',
                 source='pending', destination='running')]
        path.write_text(''.join(json.dumps(row)+'\n' for row in events))
        with mock.patch.object(gate.old, 'CENTRAL_AUDIT', path):
            batch = gate._attempt_index({jid})
            self.assertEqual(batch[jid, 1], gate.old._attempt_audit(jid, 1))
            with path.open('a') as stream:
                stream.write(json.dumps(events[0])+'\n')
            with self.assertRaisesRegex(ValueError, 'duplicate central'):
                gate._attempt_index({jid})


class RecoveryCollectorTests(unittest.TestCase):
    """Real request/marker/summary/producer gates; model semantics are a boundary fixture."""
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/model-signal-validation/recovery-collector')
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        for name, relative in dict(OUTBOX_ROOT='outbox', CENTRAL_JOBS='central/jobs',
                CENTRAL_LOGS='central/logs', CENTRAL_RETRIES='central/retries', CENTRAL_AUDIT='central/audit.jsonl').items():
            self.stack.enter_context(mock.patch.object(gate.old, name, self.root / relative))
        self.stack.enter_context(mock.patch.object(gate, 'REPORT_ROOT', self.root / 'reports'))
        self.stack.enter_context(mock.patch.object(gate, 'PROOF_ROOT', self.root / 'proofs'))
        self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=_provenance()))
        source = self.root / 'source.py'
        source.write_text('# reviewed fixture\n')
        self.sources = {'source': gate._binding(source)}
        self.stack.enter_context(mock.patch.object(builder, 'review_sources', return_value=self.sources))
        owner = self

        class FixtureStore:
            def __init__(self, config, provenance, prerequisites):
                self.config, self.provenance, self.prerequisites = config, provenance, prerequisites
                self.run_id, self.config_sha256 = config['run_id'], gate.digest(config)
                self.directory = owner.root / 'outputs' / self.run_id
            def _current(self):
                owner.assertEqual(self.provenance, _provenance())
            def _manifest(self):
                owner.assertEqual(gate._read_json(self.directory/'run_manifest.env'), self.config)
            def identity(self, seed, split):
                return dict(run_id=self.run_id, config_sha256=self.config_sha256,
                    provenance_sha256=self.provenance['provenance_sha256'], seed=seed, split=split,
                    cell=self.config['cell'], variant=self.config['variant'])
            def cell_dir(self, seed, split):
                return self.directory / ('seed_%03d_%s' % (seed, split))
            def _started(self, seed, split):
                row = gate._read_json(self.cell_dir(seed, split)/'started.json')
                if row['identity'] != self.identity(seed, split) or row['prerequisites'] != self.prerequisites:
                    raise ValueError('fixture started identity/prerequisites mismatch')
                gate.old.validate_execution(row['execution'])
                return row
            def _read(self, seed, split):
                return gate._read_json(self.cell_dir(seed, split)/'result.json'), {}
            def _read_noncompletion(self, seed, split):
                return gate._read_json(self.cell_dir(seed, split)/'noncompletion.json'), {}

        self.stack.enter_context(mock.patch.object(recovery, 'RecoveryStore', FixtureStore))
        self.requests, audit, references, controls = [], [], [], []
        prerequisites = dict(closeout=protocol.SOURCE_CLOSEOUT, pilot=None)
        self.directories = {}
        for spec in builder.build_requests():
            request, config = spec['request'], spec['configuration']
            jid, p = request['job_id'], request['parameters']
            path = gate.old.OUTBOX_ROOT / spec['filename']
            _write(path, request)
            # Builder bytes are canonical JSON, rather than pretty fixture serialization.
            path.write_bytes(builder.encoded(request))
            self.requests.append(dict(path=str(path), sha256=_hash(path), job_id=jid,
                task=request['task'], parameters=p, run_id=spec['run_id'], configuration=config,
                configuration_sha256=spec['configuration_sha256'], seeds=spec['seed_sequence'], splits=spec['splits']))
            store = FixtureStore(config, _provenance(), prerequisites)
            _write(store.directory/'run_manifest.env', config)
            terminal_units = []
            for seed in spec['seed_sequence']:
                for split in protocol.SPLITS:
                    key = p['setting'], p['cell'], p['variant'], seed, split
                    record = _result(key)
                    capped = seed == 25 and split == 'noniid'
                    directory = store.cell_dir(seed, split)
                    self.directories[key] = directory
                    _write(directory/'started.json', dict(identity=store.identity(seed, split),
                        execution=_receipt(jid), prerequisites=prerequisites))
                    files = recovery.NONCOMPLETE_FILES if capped else recovery.COMPLETE_FILES
                    _write(directory/files[2], dict(record, identity=store.identity(seed, split)))
                    (directory/files[1]).write_bytes(b'model semantics are tested separately')
                    marker = dict(identity=store.identity(seed, split), artifacts={name: _hash(directory/name) for name in files[:-1]})
                    if capped:
                        marker['outcome'] = 'validation_patience_limit_200'
                    _write(directory/files[-1], marker)
                    terminal_units.append(dict(seed=seed, split=split,
                        outcome='budget_exhausted' if capped else 'complete',
                        artifacts={name: _hash(directory/name) for name in files}))
                    if seed == 0 or (seed, split) in ((9, 'iid'), (9, 'noniid'), (25, 'iid')):
                        reference_path = self.root / 'historical' / config['run_id'] / ('%s-%s.json' % (seed, split)) / 'result.json'
                        _write(reference_path, record)
                        ref = dict(zip(('setting', 'cell', 'variant', 'seed', 'split'), key),
                                   artifacts=[gate._binding(reference_path)])
                        (references if seed == 0 else controls).append(ref)
            execution = _receipt(jid)
            job = dict(request, state='completed', exit_code=0,
                       **{k: v for k, v in execution.items() if k not in ('job_id', 'attempt')})
            for kind in ('stdout', 'stderr'):
                log = gate.old.CENTRAL_LOGS / (jid+'.attempt-001.'+kind+'.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text('')
                job[kind+'_log'] = str(log)
            _write(gate.old.CENTRAL_JOBS/(jid+'.json'), job)
            _write(gate.old.CENTRAL_RETRIES/(jid+'.json'), dict(schema_version=2, job_id=jid,
                attempts_started=1, max_attempts=3, next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
            summary = recovery._summary_body(store, job, execution, verified_units=terminal_units)
            _write(store.directory/'shard_summaries'/(jid+'.json'), summary)
            audit.extend([dict(schema_version=2, job_id=jid, event='job_attempt_started', attempt=1),
                dict(schema_version=2, job_id=jid, event='job_transition', project='FedFisher', source='pending', destination='running')])
        gate.old.CENTRAL_AUDIT.write_text(''.join(json.dumps(row)+'\n' for row in audit))
        closeout = dict(binding=protocol.SOURCE_CLOSEOUT, old_pilot=dict(units=references),
            proof=dict(known_success_diagnostic_controls=controls), evidence_snapshot=mock.Mock())
        self.stack.enter_context(mock.patch.object(gate, 'verify_closeout', return_value=closeout))
        self.inventory_path = gate.REPORT_ROOT/'pilot.inventory.json'
        _write(self.inventory_path, dict(schema_version=1, bundle_ready=True, generation=1,
            provenance=_provenance(), task=protocol.PILOT_TASK, closeout=protocol.SOURCE_CLOSEOUT,
            review_sources=self.sources, request_count=290, scientific_cell_count=588, requests=self.requests))

    def test_full_pilot_proof_replay_uses_summary_writer_and_rejects_self_declared_success(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        self.assertEqual((proof['validated_request_count'], proof['validated_unit_count']), (290, 588))
        self.assertEqual((proof['complete_units'], proof['noncompleted_units']), (586, 2))
        self.assertFalse(proof['whole_pilot_scientific_success'])
        self.assertEqual(len(proof['shards']), 290)
        self.assertTrue(all(row['summary_producer']['producer_attempt'] == 1 for row in proof['shards']))
        path = gate.default_acceptance_path()
        _write(path, proof)
        self.assertEqual(gate.verify_pilot_acceptance(path), proof)
        _write(path, dict(proof, complete_units=588, noncompleted_units=0))
        with self.assertRaisesRegex(ValueError, 'freshly verified'):
            gate.verify_pilot_acceptance(path)

    def test_capped_unit_mixed_files_or_incomplete_summary_fail_closed(self):
        key = ('small_mlp', 'fedavg_small_e1_patience', 'site_gaussian_shift', 25, 'noniid')
        directory = self.directories[key]
        _write(directory/'done.json', dict(passed=True))
        with self.assertRaisesRegex(ValueError, 'partial or mixed'):
            gate.collect_pilot_acceptance(self.inventory_path)
        (directory/'done.json').unlink()
        summary_path = next((directory.parent/'shard_summaries').iterdir())
        summary = gate._read_json(summary_path)
        summary['units'] = summary['units'][:-1]
        _write(summary_path, summary)
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_rehashed_science_tampering_is_rejected(self):
        key = ('small_mlp', 'fedavg_small_e1_patience', 'site_gaussian_shift', 9, 'iid')
        directory = self.directories[key]
        result = gate._read_json(directory/'result.json')
        result['fit']['numeric_parameter'] += .01
        _write(directory/'result.json', result)
        marker = gate._read_json(directory/'done.json')
        marker['artifacts']['result.json'] = _hash(directory/'result.json')
        _write(directory/'done.json', marker)
        with self.assertRaisesRegex(ValueError, 'numeric fitting path'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_source_race_bad_schema_and_unsupported_cap_marker_are_rejected(self):
        with mock.patch.object(gate, 'current_provenance', side_effect=[_provenance(), dict(_provenance(), git_dirty='1')]):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                gate.collect_pilot_acceptance(self.inventory_path)
        inventory = gate._read_json(self.inventory_path)
        _write(self.inventory_path, dict(inventory, schema_version=2))
        with self.assertRaisesRegex(ValueError, 'reviewed current source'):
            gate.collect_pilot_acceptance(self.inventory_path)
        _write(self.inventory_path, inventory)
        key = ('small_mlp', 'fedavg_small_e1_patience', 'site_gaussian_shift', 25, 'noniid')
        path = self.directories[key]/'noncompletion_marker.json'
        marker = gate._read_json(path)
        _write(path, dict(marker, outcome='arbitrary_unknown_failure'))
        with self.assertRaisesRegex(ValueError, 'unsupported scientific noncompletion'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_summary_writer_must_match_a_real_producing_attempt(self):
        directory = next(iter(self.directories.values()))
        path = next((directory.parent/'shard_summaries').iterdir())
        summary = gate._read_json(path)
        summary['producer_execution']['estimated_runtime_seconds'] = 21.
        _write(path, summary)
        with self.assertRaisesRegex(ValueError, 'producing-attempt receipt'):
            gate.collect_pilot_acceptance(self.inventory_path)


if __name__ == '__main__':
    unittest.main()
