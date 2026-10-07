"""Real producer/evidence and immutable acceptance checks on scratch fixtures."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure.requests import encoded
from effect_modifier_experiment.survival import acceptance as old_gate
from effect_modifier_experiment.survival_pooled import acceptance as pooled_gate
from effect_modifier_experiment.survival_comparators import acceptance as gate, protocol
from effect_modifier_experiment.survival_comparators.requests import build_requests


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComparatorAcceptanceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/survival-comparator-validation/acceptance')
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.source = dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                           source_tree_sha256='c' * 64, provenance_sha256='d' * 64)
        for name, value in dict(OUTBOX_ROOT=self.root/'outbox', INVENTORY_ROOT=self.root/'reports',
                               PROOF_ROOT=self.root/'proofs', CENTRAL_LOGS=self.root/'central/logs',
                               CENTRAL_RETRIES=self.root/'central/retries').items():
            self.stack.enter_context(mock.patch.object(gate, name, value))
            if name.startswith('CENTRAL_'):
                self.stack.enter_context(mock.patch.object(old_gate, name, value))
        self.jobs = self.root/'central/jobs'
        self.audit = self.root/'central/audit.jsonl'
        self.stack.enter_context(mock.patch.object(pooled_gate, 'CENTRAL_JOBS', self.jobs))
        self.stack.enter_context(mock.patch.object(old_gate, 'CENTRAL_AUDIT', self.audit))
        self.stack.enter_context(mock.patch.object(gate, 'current_provenance', return_value=self.source))
        self.stack.enter_context(mock.patch.object(gate, 'validate_inputs'))
        self.sources = {'fixture': {'path': 'source', 'sha256': '1' * 64}}
        self.stack.enter_context(mock.patch.object(gate, 'review_sources', return_value=self.sources))
        specs, requests, audit = build_requests(), [], []
        for spec in specs:
            raw, cfg = spec['request'], spec['configuration']
            path = gate.OUTBOX_ROOT/spec['filename']
            write_json(path, raw)
            requests.append(dict(path=str(path), sha256=spec['request_sha256'],
                **{k: raw[k] for k in ('job_id', 'task', 'parameters')},
                configuration=cfg, configuration_sha256=spec['configuration_sha256'],
                run_id=spec['run_id'], seeds=spec['seed_sequence']))
            seed = raw['parameters']['seed']
            run = self.root/'outputs'/cfg['run_id']
            write_json(run/'run_manifest.env', {'fixture': cfg})
            cell = run/('seed_%03d' % seed)
            execution = dict(job_id=raw['job_id'], attempt=1, execution_profile='cpu-1core',
                estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
                    gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))
            write_json(cell/'started.json', dict(execution=execution))
            result = dict(site_sizes=[500, 501], fingerprints=dict(training=cfg['scenario'],
                evaluation='test-%d' % seed, evaluation_truth='truth-%d' % seed),
                references={'fixture': True}, fit_origin=cfg['fit_origin'], prediction={}, timing={}, memory={},
                discovery={tag: dict(raw_x_sha256='x%d' % seed, contrast_truth_sha256='truth%d' % seed,
                                     permutation_sha256='perm%d' % seed) for tag in protocol.TAGS})
            write_json(cell/'result.json', result)
            (cell/'model.npz').write_bytes(b'numerical validation boundary covered by separate actual-fit fixtures')
            self.rehash(cell)
            job = dict(raw, state='completed', exit_code=0,
                       **{k: v for k, v in execution.items() if k not in ('job_id', 'attempt')})
            for kind in ('stdout', 'stderr'):
                log = gate.CENTRAL_LOGS/('%s.attempt-001.%s.log' % (raw['job_id'], kind))
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text('')
                job[kind + '_log'] = str(log)
            write_json(self.jobs/(raw['job_id'] + '.json'), job)
            write_json(gate.CENTRAL_RETRIES/(raw['job_id'] + '.json'),
                dict(schema_version=2, job_id=raw['job_id'], attempts_started=1, max_attempts=3,
                     next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
            audit.extend([dict(schema_version=2, job_id=raw['job_id'], event='job_attempt_started', attempt=1),
                          dict(schema_version=2, job_id=raw['job_id'], event='job_transition', project='FedFisher',
                               source='pending', destination='running')])
        self.audit.write_text(''.join(json.dumps(row) + '\n' for row in audit))
        self.inventory = dict(schema_version=1, bundle='fedfisher-survival-comparators-v1-10seed',
            generation=1, task=protocol.PILOT_TASK, request_count=20, scientific_cell_count=20,
            fitted_model_count=20, replayed_model_count=20, formal_expansion_authorized=False,
            coverage=dict(scenarios=list(protocol.SCENARIOS), seeds=list(protocol.SEEDS), models=list(protocol.TAGS)),
            provenance=self.source, project_head=self.source['git_commit'], worktree_status='clean', bundle_ready=True,
            historical_inputs=gate.historical_binding(), review_sources=self.sources, requests=requests)
        self.inventory_path = gate.INVENTORY_ROOT/'fixture.inventory.json'
        write_json(self.inventory_path, self.inventory)
        owner = self

        class NumericalBoundary:
            artifact_names = ('started.json', 'model.npz', 'result.json')
            result_name, done_name = 'result.json', 'done.json'

            def __init__(self, config, source):
                owner.assertEqual(source, owner.source)
                self.run_id = config['run_id']
                self.directory = owner.root/'outputs'/self.run_id

            def cell_dir(self, seed):
                return self.directory/('seed_%03d' % seed)

            def status(self, seed):
                cell = self.cell_dir(seed)
                if not (cell/'done.json').exists():
                    return 'partial'
                done = json.loads((cell/'done.json').read_text())
                if any(checksum(cell/name) != value for name, value in done.items()):
                    raise ValueError('fixture artifact checksum changed')
                return 'validated'

        self.stack.enter_context(mock.patch.object(gate, 'ArtifactStore', NumericalBoundary))

    def rehash(self, cell):
        write_json(cell/'done.json', {name: checksum(cell/name) for name in ('started.json', 'model.npz', 'result.json')})

    def cell(self, scenario='iid', seed=0):
        return self.root/'outputs'/('survival_comparators_v1_%s_10seeds' % scenario)/('seed_%03d' % seed)

    def test_complete_evidence_is_saved_only_and_proof_is_immutable(self):
        with mock.patch('effect_modifier_experiment.survival_mlp8.methods.MLP8Fit.predict_eta', side_effect=AssertionError('inference')):
            proof = gate.collect_acceptance(self.inventory_path)
        self.assertEqual(proof['validated_unit_count'], 20)
        self.assertEqual(proof['paired_checks']['complete_seed_count'], 10)
        self.assertFalse(proof['inference_performed'])
        path = gate.publish_acceptance(proof)
        self.assertEqual(json.loads(path.read_bytes()), proof)
        with self.assertRaises(ValueError):
            gate.publish_acceptance(proof)

    def test_incomplete_inventory_partial_artifact_and_corrupted_hash_are_rejected(self):
        altered = deepcopy(self.inventory)
        altered['requests'].pop()
        write_json(self.inventory_path, altered)
        with self.assertRaises(ValueError):
            gate.collect_acceptance(self.inventory_path)
        write_json(self.inventory_path, self.inventory)
        cell = self.cell()
        marker = (cell/'done.json').read_bytes()
        (cell/'done.json').unlink()
        with self.assertRaises(ValueError):
            gate.collect_acceptance(self.inventory_path)
        (cell/'done.json').write_bytes(marker)
        (cell/'model.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            gate.collect_acceptance(self.inventory_path)

    def test_failed_central_attempt_and_forged_producing_receipt_are_rejected(self):
        job_path = self.jobs/(self.inventory['requests'][0]['job_id'] + '.json')
        job = json.loads(job_path.read_bytes())
        write_json(job_path, dict(job, state='failed', exit_code=1))
        with self.assertRaises(ValueError):
            gate.collect_acceptance(self.inventory_path)
        write_json(job_path, job)
        cell = self.cell()
        started = json.loads((cell/'started.json').read_bytes())
        started['execution']['attempt'] = 2
        write_json(cell/'started.json', started)
        self.rehash(cell)
        with self.assertRaises(ValueError):
            gate.collect_acceptance(self.inventory_path)

    def test_rehashed_cross_scenario_truth_and_permutation_changes_are_rejected(self):
        cell = self.cell('covariate_shift')
        result = json.loads((cell/'result.json').read_bytes())
        for mutate in (lambda r: r['fingerprints'].update(evaluation_truth='wrong'),
                       lambda r: r['discovery']['fedavg'].update(permutation_sha256='wrong')):
            changed = deepcopy(result)
            mutate(changed)
            write_json(cell/'result.json', changed)
            self.rehash(cell)
            with self.assertRaises(ValueError):
                gate.collect_acceptance(self.inventory_path)

    def test_source_drift_and_wrong_scientific_request_fail_closed(self):
        altered = deepcopy(self.inventory)
        altered['requests'][0]['parameters']['seed'] = 10
        write_json(self.inventory_path, altered)
        with self.assertRaises(ValueError):
            gate.collect_acceptance(self.inventory_path)
        write_json(self.inventory_path, self.inventory)
        proof = gate.collect_acceptance(self.inventory_path)
        with mock.patch.object(gate, 'current_provenance', return_value=dict(self.source, git_commit='e' * 40)):
            with self.assertRaises(ValueError):
                gate.publish_acceptance(proof)


if __name__ == '__main__':
    unittest.main()
