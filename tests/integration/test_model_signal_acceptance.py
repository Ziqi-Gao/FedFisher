"""Fabricated saved evidence only; artifact semantics have their own fixtures."""
from tests.paths import PROJECT_ROOT
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.model_signal_ablation import acceptance as gate, protocol


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def provenance():
    return dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                source_tree_sha256='c' * 64, provenance_sha256='d' * 64)


def execution(job_id, attempt=1):
    return dict(job_id=job_id, attempt=attempt, execution_profile='cpu-1core',
        estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
        gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def result(setting, cell, variant, split):
    fingerprint = lambda value: hashlib.sha256(value.encode()).hexdigest()
    data = setting + split + ('shift' if 'site_gaussian' in variant else 'control')
    local = 'local_permutation' in variant
    method = protocol.CELLS[cell]
    fit = dict(fit_fingerprint=fingerprint(data + cell))
    if method['family'] == 'legacy_fedfisher':
        fit = dict(initial_model_fingerprint=fingerprint(method['model']),
            local_model_fingerprints=[fingerprint(data + method['model'])], validation={'fixed': data})
    return dict(status='complete', site_sizes=[500, 500], label_counts={'0': {'0': 500}, '1': {'1': 500}},
        fingerprints=dict(training=fingerprint(data), test=fingerprint(setting + split),
                          model=fingerprint(data + cell)), fit=fit,
        evaluation=dict(raw_x_sha256=fingerprint(setting + split + 'x'),
            permutation_sha256=fingerprint(setting + split + str(local)),
            shard_indices_sha256=[fingerprint(str(local))], shard_sizes=[10000]),
        timing={'measured_fixture': True}, memory=dict(process_peak_rss_kib=1000))


def paired_records():
    return {(setting, cell, variant, split): result(setting, cell, variant, split)
        for setting, recipe in protocol.SETTINGS.items() for cell in recipe['cells']
        for variant in protocol.VARIANTS for split in protocol.SPLITS}


class ModelSignalPairTests(unittest.TestCase):
    def test_complete_mixed_family_scope_allows_changed_outcomes_between_signals(self):
        proof = gate.validate_pairs(paired_records())
        self.assertEqual(proof, dict(intervention_only_pairs=288, method_data_pairs=552,
            diag_kfac_local_fit_pairs=56, common_test_sets=6, passed=True))

    def test_missing_scope_changed_data_fit_and_fisher_local_models_fail(self):
        mutations = [
            (('weak_signal_1', 'dslassocov', 'control', 'iid'),
             lambda r: r['fingerprints'].update(training='x')),
            (('small_mlp', 'fedavg_small_e1_budget30', 'local_permutation', 'iid'),
             lambda r: r['fit'].update(fit_fingerprint='x')),
            (('small_mlp', 'fedfisher_kfac_small', 'control', 'iid'),
             lambda r: r['fit'].update(local_model_fingerprints=['x'])),
            (('weak_signal_05', 'dslassocov', 'site_gaussian_shift', 'noniid'),
             lambda r: r['evaluation'].update(permutation_sha256='x')),
        ]
        for key, change in mutations:
            records = paired_records()
            change(records[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.validate_pairs(records)
        records.pop(next(iter(records)))
        with self.assertRaisesRegex(ValueError, 'exactly all'):
            gate.validate_pairs(records)


class ModelSignalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/model-signal-validation/acceptance')
        gate._ordinary_path(scratch, exists=False)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        for name, relative in dict(OUTBOX_ROOT='outbox', REPORT_ROOT='reports', CENTRAL_JOBS='central/jobs',
                CENTRAL_LOGS='central/logs', CENTRAL_RETRIES='central/retries', CENTRAL_AUDIT='central/audit.jsonl',
                PROJECT_ROOT='source').items():
            self.stack.enter_context(mock.patch.object(gate, name, self.root / relative))
        self.current = self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
        source = gate.PROJECT_ROOT / 'protocol.py'
        source.parent.mkdir(parents=True)
        source.write_text('# fabricated reviewed protocol\n')
        self.directories, requests, audit = {}, [], []
        for setting, recipe in protocol.SETTINGS.items():
            for cell in recipe['cells']:
                for variant in protocol.VARIANTS:
                    p = dict(setting=setting, cell=cell, variant=variant, shard_index=0, shard_count=1)
                    config = protocol.run_config(protocol.PILOT_TASK, p)
                    job_id = 'fedfisher-ms-fixture-%03d' % len(requests)
                    path = gate.OUTBOX_ROOT / 'bundle' / (job_id + '.job-request.json')
                    request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                                   task=protocol.PILOT_TASK, parameters=p, priority=0)
                    write_json(path, request)
                    requests.append(dict(path=str(path), sha256=checksum(path), job_id=job_id,
                        task=protocol.PILOT_TASK, parameters=p, seeds=[0], splits=list(protocol.SPLITS),
                        run_id=config['run_id'], configuration=config, configuration_sha256=gate.digest(config)))
                    for split in protocol.SPLITS:
                        directory = self.root / 'outputs' / config['run_id'] / ('seed_000_' + split)
                        directory.mkdir(parents=True)
                        self.directories[setting, cell, variant, split] = directory
                        write_json(directory.parent / 'run_manifest.env', {'fixture': config})
                        write_json(directory / 'started.json', dict(execution=execution(job_id)))
                        write_json(directory / 'result.json', result(setting, cell, variant, split))
                        (directory / 'model.npz').write_bytes(b'fabricated model; store boundary mocked')
                        self.rehash(directory)
                    job = dict(request, state='completed', exit_code=0,
                        **{k: v for k, v in execution(job_id).items() if k not in ('job_id', 'attempt')})
                    for kind in ('stdout', 'stderr'):
                        log = gate.CENTRAL_LOGS / (job_id + '.attempt-001.' + kind + '.log')
                        log.parent.mkdir(parents=True, exist_ok=True)
                        log.write_text('')
                        job[kind + '_log'] = str(log)
                    write_json(gate.CENTRAL_JOBS / (job_id + '.json'), job)
                    write_json(gate.CENTRAL_RETRIES / (job_id + '.json'), dict(schema_version=2, job_id=job_id,
                        attempts_started=1, max_attempts=3, next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
                    audit.extend([dict(schema_version=2, job_id=job_id, event='job_attempt_started', attempt=1),
                        dict(schema_version=2, job_id=job_id, event='job_transition', project='FedFisher',
                             source='pending', destination='running')])
        gate.CENTRAL_AUDIT.write_text(''.join(json.dumps(item) + '\n' for item in audit))
        self.inventory = dict(schema_version=1, task=protocol.PILOT_TASK, request_count=288,
            scientific_cell_count=576, provenance=provenance(), bundle_ready=True,
            review_sources={'protocol': gate._binding(source)}, requests=requests)
        self.path = gate.REPORT_ROOT / 'pilot.inventory.json'
        write_json(self.path, self.inventory)
        owner = self
        class FabricatedStore:
            def __init__(self, config, source_provenance):
                owner.assertEqual(source_provenance, provenance())
                self.key = config['setting'], config['cell'], config['variant']
            def cell_dir(self, seed, split):
                owner.assertEqual(seed, 0)
                return owner.directories[(*self.key, split)]
            def status(self, seed, split):
                directory = self.cell_dir(seed, split)
                done = json.loads((directory / 'done.json').read_text())
                if any(checksum(directory / name) != expected for name, expected in done.items()):
                    raise ValueError('fabricated artifact checksum mismatch')
                return 'validated'
        self.stack.enter_context(mock.patch.object(gate, 'ArtifactStore', FabricatedStore))

    def rehash(self, directory):
        write_json(directory / 'done.json', {name: checksum(directory / name)
            for name in ('started.json', 'model.npz', 'result.json')})

    def test_complete_acceptance_replay_retry_and_tampering(self):
        proof = gate.collect_pilot_acceptance(self.path)
        self.assertEqual((proof['validated_request_count'], proof['validated_unit_count']), (288, 576))
        self.assertEqual(proof['formal_scope']['scientific_units'], 172800)
        self.assertFalse(proof['inference_performed'])
        write_json(gate.default_acceptance_path(), proof)
        self.assertEqual(gate.check_current_pilot_gate(), proof)
        altered = deepcopy(proof)
        altered['validated_unit_count'] = 1
        write_json(gate.default_acceptance_path(), altered)
        with self.assertRaisesRegex(ValueError, 'freshly verified'):
            gate.check_current_pilot_gate()
        # A final skip attempt must preserve the actual producer's measured time.
        request = self.inventory['requests'][0]
        job_id = request['job_id']
        path = gate.CENTRAL_JOBS / (job_id + '.json')
        job = json.loads(path.read_text())
        job['estimated_runtime_seconds'] = 90.
        for kind in ('stdout', 'stderr'):
            log = gate.CENTRAL_LOGS / (job_id + '.attempt-002.' + kind + '.log')
            log.write_text('')
            job[kind + '_log'] = str(log)
        write_json(path, job)
        retry_path = gate.CENTRAL_RETRIES / (job_id + '.json')
        retry = json.loads(retry_path.read_text())
        retry['attempts_started'] = 2
        write_json(retry_path, retry)
        resumed = gate.collect_pilot_acceptance(self.path)['units'][0]['producer_evidence']
        self.assertEqual((resumed['producer_attempt'], resumed['final_completed_attempt']), (1, 2))
        self.assertEqual(resumed['execution_receipt']['estimated_runtime_seconds'], 20.)
        # Rehashing a scientifically mismatched result cannot bypass pair checks.
        directory = self.directories['weak_signal_1', 'dslassocov', 'control', 'iid']
        record = json.loads((directory / 'result.json').read_text())
        record['fingerprints']['training'] = 'x' * 64
        write_json(directory / 'result.json', record)
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError, 'paired data'):
            gate.collect_pilot_acceptance(self.path)

    def test_missing_forged_duplicate_and_failed_evidence_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            gate.check_current_pilot_gate()
        changed = deepcopy(self.inventory)
        changed['requests'][1] = changed['requests'][0]
        write_json(self.path, changed)
        with self.assertRaisesRegex(ValueError, 'duplicate pilot'):
            gate.collect_pilot_acceptance(self.path)
        write_json(self.path, self.inventory)
        request = self.inventory['requests'][0]
        central = gate.CENTRAL_JOBS / (request['job_id'] + '.json')
        job = json.loads(central.read_text())
        write_json(central, dict(job, exit_code=False))
        with self.assertRaisesRegex(ValueError, 'matching completed'):
            gate.collect_pilot_acceptance(self.path)
        write_json(central, job)
        stderr = Path(job['stderr_log'])
        stderr.write_text('RuntimeError: unexpected failure\n')
        with self.assertRaisesRegex(ValueError, 'unexpected pilot stderr'):
            gate.collect_pilot_acceptance(self.path)
        stderr.write_text('')
        request_path = Path(request['path'])
        request_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'request bytes changed'):
            gate.collect_pilot_acceptance(self.path)

    def test_cli_publishes_canonical_proof_exclusively_and_preserves_source_race(self):
        source = PROJECT_ROOT / 'scripts/validate_model_signal_pilot.py'
        spec = importlib.util.spec_from_file_location('validate_model_signal_pilot_fixture', source)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        proof = gate.collect_pilot_acceptance(self.path)
        output = gate.default_acceptance_path()
        self.current.side_effect = [provenance(), provenance(), dict(provenance(), git_dirty='1')]
        with mock.patch.object(gate, 'collect_pilot_acceptance', return_value=proof):
            with self.assertRaisesRegex(ValueError, 'partial proof preserved'):
                cli.main(['--inventory', str(self.path)])
        self.assertFalse(output.exists())
        pending = output.with_name('.' + output.name + '.pending')
        self.assertTrue(pending.exists())
        self.current.side_effect = None
        with self.assertRaises(FileExistsError):
            cli.main(['--inventory', str(self.path)])


if __name__ == '__main__':
    unittest.main()
