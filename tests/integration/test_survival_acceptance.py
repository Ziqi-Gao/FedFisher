"""Fabricated acceptance evidence only: no generation, model fitting or inference.

ArtifactStore's numerical semantics have their independent contract suite.
These tests replace that boundary with a hash-checking fabricated store, and
exercise the new inventory, pairing, central evidence and proof-replay logic.
"""
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

import numpy as np

from effect_modifier_experiment.survival import acceptance as gate, protocol


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    return dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                source_tree_sha256='c' * 64, provenance_sha256='d' * 64)


def execution(job_id, attempt=1):
    return dict(job_id=job_id, attempt=attempt, execution_profile='cpu-1core',
                estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
                gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


class SurvivalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-validation/acceptance')
        gate._ordinary_path(scratch, exists=False)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        locations = dict(OUTBOX_ROOT=self.root / 'outbox', INVENTORY_ROOT=self.root / 'reports',
                         PROOF_ROOT=self.root / 'proofs', CENTRAL_JOBS=self.root / 'central/jobs',
                         CENTRAL_LOGS=self.root / 'central/logs', CENTRAL_RETRIES=self.root / 'central/retries',
                         CENTRAL_AUDIT=self.root / 'central/audit.jsonl')
        for key, value in locations.items():
            self.stack.enter_context(mock.patch.object(gate, key, value))
        self.source = self.root / 'source/protocol.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_text('# fabricated reviewed source\n')
        self.stack.enter_context(mock.patch.object(gate, 'PROJECT_ROOT', self.source.parent))
        self.current = self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=provenance()))
        self.directories, requests, audit = {}, [], []
        self.status_calls = []
        for cell in protocol.CELLS:
            for scenario in protocol.SCENARIOS:
                parameters = dict(cell=cell, scenario=scenario)
                config = protocol.run_config(protocol.PILOT_TASK, parameters)
                job_id = 'fedfisher-survival-fixture-' + cell + '-' + scenario
                request_path = gate.OUTBOX_ROOT / 'bundle' / (job_id + '.job-request.json')
                request = dict(schema_version=2, job_id=job_id, project='FedFisher',
                               task=protocol.PILOT_TASK, parameters=parameters, priority=0)
                write_json(request_path, request)
                requests.append(dict(path=str(request_path), sha256=checksum(request_path), job_id=job_id,
                    task=protocol.PILOT_TASK, parameters=parameters, seeds=[0], run_id=config['run_id'],
                    configuration=config, configuration_sha256=hashlib.sha256(gate.canonical(config).encode()).hexdigest()))
                directory = self.root / 'outputs' / config['run_id'] / 'seed_000'
                directory.mkdir(parents=True)
                self.directories[cell, scenario] = directory
                write_json(directory.parent / 'run_manifest.env', {'fixture_config': config})
                write_json(directory / 'started.json', dict(execution=execution(job_id)))
                write_json(directory / 'result.json', self.result(cell, scenario))
                arrays = {'vector': np.array([0.1, 0.2])}
                if cell != 'dslasso':
                    depth = 1 if cell.endswith('shallow') else 2
                    for index in range(10):
                        stream = io.BytesIO()
                        np.savez(stream, vector=np.array([index, depth], dtype=float))
                        arrays['upload_%03d' % index] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                np.savez(directory / 'model.npz', **arrays)
                self.rehash(directory)
                job = dict(request, state='completed', exit_code=0, **{
                    key: value for key, value in execution(job_id).items() if key not in ('job_id', 'attempt')})
                for kind in ('stdout', 'stderr'):
                    path = gate.CENTRAL_LOGS / ('%s.attempt-001.%s.log' % (job_id, kind))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('')
                    job[kind + '_log'] = str(path)
                write_json(gate.CENTRAL_JOBS / (job_id + '.json'), job)
                write_json(gate.CENTRAL_RETRIES / (job_id + '.json'), dict(schema_version=2, job_id=job_id,
                    attempts_started=1, max_attempts=3, next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
                audit.extend([dict(schema_version=2, job_id=job_id, event='job_attempt_started', attempt=1),
                    dict(schema_version=2, job_id=job_id, event='job_transition', project='FedFisher',
                         source='pending', destination='running')])
        gate.CENTRAL_AUDIT.write_text(''.join(json.dumps(record) + '\n' for record in audit))
        self.inventory = dict(schema_version=1, bundle='fedfisher-survival-v1-pilot', generation=1,
            task=protocol.PILOT_TASK, request_count=10, scientific_cell_count=10,
            coverage=dict(method_cells=5, scenarios=2, seeds=[0]), provenance=provenance(),
            project_head=provenance()['git_commit'], worktree_status='clean', bundle_ready=True,
            submitted=False, review_sources=dict(protocol=gate._binding(self.source)), requests=requests)
        self.inventory_path = gate.INVENTORY_ROOT / 'pilot.inventory.json'
        write_json(self.inventory_path, self.inventory)
        owner = self

        class FabricatedStore:
            def __init__(self, config, source):
                owner.assertEqual(source, provenance())
                self.key = config['cell'], config['scenario']

            def cell_dir(self):
                return owner.directories[self.key]

            def status(self):
                owner.status_calls.append(self.key)
                directory = self.cell_dir()
                done = json.loads((directory / 'done.json').read_text())
                if any(checksum(directory / name) != digest for name, digest in done.items()):
                    raise ValueError('fabricated artifact checksum mismatch')
                return 'validated'

        self.store_factory = self.stack.enter_context(mock.patch.object(gate, 'ArtifactStore', FabricatedStore))

    def result(self, cell, scenario):
        sizes = list(range(500, 510))
        diagnostics = [dict(site_id=i, tau_beyond_observed_max=False, n_at_risk_tau=30,
                            all_zero=False, all_one=False) for i in range(10)]
        evaluation = dict(raw_x_sha256='1' * 64, contrast_truth_sha256='2' * 64,
            permutation_sha256='3' * 64, recovery=dict(probability_all_signals_selected=0., auroc=0.1),
            contrast_rmse=10.)  # Deliberately poor; no invented performance gate.
        local = [dict(site_id=i, sample_count=n, optimizer_updates=30, epochs=30,
                      stop_reason='fixed_epoch_budget', initial_loss=2., final_loss=1.,
                      pseudo_seconds=.01, training_seconds=.1, curvature_seconds=.1,
                      serialization_seconds=.01) for i, n in enumerate(sizes)]
        if cell == 'dslasso':
            fit = dict(timing_seconds=dict(cross_validation=1., refit=1., full_fit=2.),
                       cv_paths=[dict(fold=i, lambda_max=0. if i == 0 else .1,
                                     fits=[dict(converged=True)]) for i in range(5)])
        else:
            fit = dict(initial_sha256=('4' if cell.endswith('shallow') else '5') * 64,
                       local=local, total_fit_seconds=2., server=dict(aggregation_seconds=.1))
        return dict(status='complete', site_sizes=sizes, pseudo_diagnostics=diagnostics,
            fingerprints=dict(training=('6' if scenario == 'iid' else '7') * 64,
                              evaluation='8' * 64, evaluation_truth='9' * 64, model='0' * 64),
            fit=fit, evaluation=evaluation,
            timing=dict(stages={stage: dict(wall_seconds=3., cpu_seconds=2.)
                                for stage in ('data_preparation', 'fit', 'evaluation')},
                        fit_includes_all_selection_and_refits=True, network_transport_measured=False),
            memory=dict(process_peak_rss_kib=1000, scope='fabricated test observation'))

    def rehash(self, directory):
        write_json(directory / 'done.json', {name: checksum(directory / name)
                   for name in ('started.json', 'model.npz', 'result.json')})

    def change_result(self, key, update):
        directory = self.directories[key]
        result = json.loads((directory / 'result.json').read_text())
        update(result)
        write_json(directory / 'result.json', result)
        self.rehash(directory)

    def test_ten_actual_evidence_cells_paired_and_replayed_without_inference(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        self.assertEqual(proof['validated_unit_count'], 10)
        self.assertEqual(len(self.status_calls), 10)
        self.assertEqual(proof['paired_checks'], dict(method_data_pairs=8, diag_kfac_local_fit_pairs=4,
                                                     common_test_sets=1, passed=True))
        self.assertEqual(proof['formal_scope']['scientific_units'], 3000)
        self.assertEqual((proof['formal_scope']['seed_start'], proof['formal_scope']['seed_end'],
                          proof['formal_scope']['seed_count']), (0, 299, 300))
        self.assertFalse(proof['inference_performed'])
        self.assertTrue(all(unit['memory']['process_peak_rss_kib'] > 0 for unit in proof['units']))
        dslasso = proof['units'][0]['quality_observations']
        self.assertEqual(dslasso['contrast_rmse'], 10.)
        self.assertFalse(dslasso['performance_threshold_applied'])
        self.assertTrue(dslasso['dslasso_cv_path_diagnostics'][0]['zero_lambda_max'])
        write_json(gate.default_acceptance_path(), proof)
        self.assertEqual(gate.check_current_pilot_gate(), proof)
        self.assertEqual(len(self.status_calls), 20)

    def test_missing_or_tampered_saved_boolean_cannot_open_gate(self):
        with self.assertRaises(FileNotFoundError):
            gate.check_current_pilot_gate()
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        proof['validated_unit_count'] = 999
        write_json(gate.default_acceptance_path(), proof)
        with self.assertRaisesRegex(ValueError, 'freshly verified'):
            gate.check_current_pilot_gate()

    def test_request_checksum_and_duplicate_coverage_fail(self):
        request = self.inventory['requests'][0]
        Path(request['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'request bytes changed'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_incomplete_method_coverage_rejected(self):
        changed = deepcopy(self.inventory)
        changed['requests'][-1] = changed['requests'][0]
        write_json(self.inventory_path, changed)
        with self.assertRaisesRegex(ValueError, 'duplicate pilot'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_not_completed_wrong_owner_or_false_exit_zero_rejected(self):
        request = self.inventory['requests'][0]
        path = gate.CENTRAL_JOBS / (request['job_id'] + '.json')
        original = json.loads(path.read_text())
        for changes in (dict(state='running'), dict(project='Other'), dict(exit_code=False),
                        dict(task=protocol.FORMAL_TASK), dict(parameters=dict(cell='dslasso', scenario='wrong'))):
            write_json(path, dict(original, **changes))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'matching completed'):
                gate.collect_pilot_acceptance(self.inventory_path)

    def test_retry_accepts_earlier_producer_without_reassigning_measured_time(self):
        request = self.inventory['requests'][0]
        job_id = request['job_id']
        path = gate.CENTRAL_JOBS / (job_id + '.json')
        job = json.loads(path.read_text())
        job['estimated_runtime_seconds'] = 90.
        for kind in ('stdout', 'stderr'):
            log = gate.CENTRAL_LOGS / ('%s.attempt-002.%s.log' % (job_id, kind))
            log.write_text('')
            job[kind + '_log'] = str(log)
        write_json(path, job)
        retry_path = gate.CENTRAL_RETRIES / (job_id + '.json')
        retry = json.loads(retry_path.read_text())
        retry['attempts_started'] = 2
        write_json(retry_path, retry)
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        producer = proof['units'][0]['producer_evidence']
        self.assertEqual((producer['producer_attempt'], producer['final_completed_attempt']), (1, 2))
        self.assertTrue(producer['resumed_from_earlier_attempt'])
        self.assertEqual(producer['execution_receipt']['estimated_runtime_seconds'], 20.)
        self.assertIn('not_archived', producer['allocation_runtime_evidence'])

    def test_retry_count_and_exact_audit_line_are_required(self):
        request = self.inventory['requests'][0]
        retry_path = gate.CENTRAL_RETRIES / (request['job_id'] + '.json')
        retry = json.loads(retry_path.read_text())
        retry['attempts_started'] = 2
        write_json(retry_path, retry)
        with self.assertRaisesRegex(ValueError, 'durable retry'):
            gate.collect_pilot_acceptance(self.inventory_path)
        retry['attempts_started'] = 1
        write_json(retry_path, retry)
        lines = gate.CENTRAL_AUDIT.read_text().splitlines(True)
        gate.CENTRAL_AUDIT.write_text(''.join(lines[1:]))
        with self.assertRaisesRegex(ValueError, 'start/running audit'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_unrelated_audit_append_is_allowed_and_producer_mutation_is_not(self):
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        write_json(gate.default_acceptance_path(), proof)
        with gate.CENTRAL_AUDIT.open('a') as stream:
            stream.write(json.dumps(dict(job_id='foreign-job', event='anything')) + '\n')
        self.assertEqual(gate.check_current_pilot_gate(), proof)
        lines = gate.CENTRAL_AUDIT.read_text().splitlines(True)
        lines[0] = lines[0].replace('"attempt": 1', '"attempt": 3')
        gate.CENTRAL_AUDIT.write_text(''.join(lines))
        with self.assertRaises(ValueError):
            gate.check_current_pilot_gate()

    def test_changed_model_vectors_and_cross_method_data_fail_pairs(self):
        key = ('fedfisher_kfac_shallow', 'iid')
        directory = self.directories[key]
        with np.load(directory / 'model.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        stream = io.BytesIO()
        np.savez(stream, vector=np.array([999., 1.]))
        arrays['upload_000'] = np.frombuffer(stream.getvalue(), dtype=np.uint8)
        np.savez(directory / 'model.npz', **arrays)
        self.rehash(directory)
        with self.assertRaisesRegex(ValueError, 'paired local model vectors'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_common_test_and_pseudo_diagnostics_must_match(self):
        key = ('dslasso', 'covariate_shift')
        self.change_result(key, lambda value: value['fingerprints'].update(evaluation='x' * 64))
        with self.assertRaisesRegex(ValueError, 'paired training/evaluation/truth'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_unexplained_logs_and_path_symlinks_rejected(self):
        request = self.inventory['requests'][0]
        stderr = gate.CENTRAL_LOGS / (request['job_id'] + '.attempt-001.stderr.log')
        stderr.write_text('RuntimeWarning: this is not the accepted warning\n')
        with self.assertRaisesRegex(ValueError, 'unexpected pilot stderr'):
            gate.collect_pilot_acceptance(self.inventory_path)
        stderr.write_text('')
        alias = self.inventory_path.with_name('alias.json')
        alias.symlink_to(self.inventory_path)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            gate.collect_pilot_acceptance(alias)

    def test_exact_known_warning_allowed_but_extra_stderr_or_stdout_failure_rejected(self):
        request = self.inventory['requests'][0]
        stderr = gate.CENTRAL_LOGS / (request['job_id'] + '.attempt-001.stderr.log')
        warning = ('/scr/del6500/FedFisher/envs/fedfisher-py39/lib64/python3.9/site-packages/'
                   'networkx/utils/backends.py:135: RuntimeWarning: '
                   'networkx backend defined more than once: nx-loopback\n'
                   '  backends.update(_get_backends("networkx.backends"))\n')
        stderr.write_text(warning)
        self.assertTrue(gate.collect_pilot_acceptance(self.inventory_path)['passed'])
        stderr.write_text(warning + 'RuntimeError: unexpected failure\n')
        with self.assertRaisesRegex(ValueError, 'unexpected pilot stderr'):
            gate.collect_pilot_acceptance(self.inventory_path)
        stderr.write_text(warning)
        stdout = gate.CENTRAL_LOGS / (request['job_id'] + '.attempt-001.stdout.log')
        stdout.write_text('Traceback (most recent call last):\n')
        with self.assertRaisesRegex(ValueError, 'failure signature'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_source_change_during_acceptance_fails(self):
        self.current.side_effect = [provenance(), dict(provenance(), source_tree_sha256='f' * 64)]
        with self.assertRaisesRegex(ValueError, 'source or pilot inventory changed'):
            gate.collect_pilot_acceptance(self.inventory_path)

    def test_cli_creates_only_canonical_proof_exclusively(self):
        path = PROJECT_ROOT / 'scripts/validate_survival_pilot.py'
        spec = importlib.util.spec_from_file_location('validate_survival_pilot_fixture', path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        output = gate.default_acceptance_path()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(script.main(['--inventory', str(self.inventory_path), '--output', str(output)]), 0)
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            script.main(['--inventory', str(self.inventory_path), '--output', str(output)])
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            script.main(['--inventory', str(self.inventory_path), '--output', str(self.root / 'wrong.json')])

    def test_cli_source_race_preserves_partial_proof_without_publishing(self):
        path = PROJECT_ROOT / 'scripts/validate_survival_pilot.py'
        spec = importlib.util.spec_from_file_location('validate_survival_pilot_race_fixture', path)
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        proof = gate.collect_pilot_acceptance(self.inventory_path)
        output = gate.default_acceptance_path()
        pending = output.with_name('.' + output.name + '.pending')
        self.current.side_effect = [provenance(), provenance(), dict(provenance(), git_dirty='1')]
        with mock.patch.object(gate, 'collect_pilot_acceptance', return_value=proof):
            with self.assertRaisesRegex(ValueError, 'partial proof preserved'):
                script.main(['--inventory', str(self.inventory_path), '--output', str(output)])
        self.assertFalse(output.exists())
        self.assertTrue(pending.is_file())
        before = pending.read_bytes()
        self.current.side_effect = None
        with self.assertRaises(FileExistsError):
            script.main(['--inventory', str(self.inventory_path), '--output', str(output)])
        self.assertEqual(pending.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
