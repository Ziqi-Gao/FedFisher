"""Fixed cohort, unchanged data and paired acceptance regressions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.fedfisher_timing import protocol
from effect_modifier_experiment.fedfisher_timing.data import generate_datasets
from effect_modifier_experiment.dslassocov_benchmark import dataset_fingerprint
from effect_modifier_experiment.run_dslassocov_effect_modifier import generate_datasets as old_generate
from types import SimpleNamespace


def paired_records():
    records = {}
    for cell in protocol.CELLS:
        for variant in protocol.VARIANTS:
            data = 'shift' if 'site_gaussian' in variant else 'base'
            for split in protocol.SPLITS:
                records[cell, variant, split] = dict(
                    fingerprints=dict(training=data+split, test='test', model=cell+data+split),
                    fit=dict(initial_model_fingerprint='initial', local_model_fingerprints=[data+split],
                             validation={'data':data+split}),
                    evaluation=dict(raw_x_sha256='raw', permutation_sha256=variant+split,
                        shard_indices_sha256=[variant+split], shard_sizes=[100]))
    return records


class TimingProtocolTests(unittest.TestCase):
    def test_report_fit_sum_does_not_double_count_components(self):
        from scripts.report_fedfisher_timing import timing_row
        result = dict(identity=dict(cell='fedfisher_diag_shallow',variant='control',
            seed=0,split='noniid',run_id='fixture'), timing=dict(stages={
                stage:dict(wall_seconds=value,cpu_seconds=value/2,components={
                    'nested':dict(wall_seconds=100,cpu_seconds=50)})
                for stage,value in zip(('data_preparation','train','tuning','aggregation','evaluation'),
                                       (2,10,0,3,4))}),
            evaluation=dict(metrics={'permute/tau_prob':dict(recovery=dict(
                expected_topk_precision=.5,probability_all_signals_selected=.25,auroc=.8))}))
        row = timing_row(result)
        self.assertEqual(row['fit_wall_seconds'],13)
        self.assertEqual(row['fit_cpu_seconds'],6.5)
        self.assertEqual(row['evaluation_wall_seconds'],4)
        self.assertEqual(row['probability_all_signals_selected'],.25)

    def test_full_deterministic_coverage_and_namespaces(self):
        for count in (8, 16):
            seeds = []
            for index in range(count):
                p = dict(cell='fedfisher_diag_shallow', variant='control', shard_index=index, shard_count=count)
                seeds.extend(protocol.seed_sequence(protocol.FORMAL_TASK, p))
            self.assertEqual(sorted(seeds), list(range(300)))
        runs = set()
        for cell in protocol.CELLS:
            for variant in protocol.VARIANTS:
                p = dict(cell=cell, variant=variant, shard_index=0, shard_count=1)
                self.assertEqual(protocol.seed_sequence(protocol.PILOT_TASK, p), (0,))
                config = protocol.run_config(protocol.PILOT_TASK, p)
                self.assertEqual(config['method']['optimizer']['local_epochs'], 30)
                self.assertEqual(config['evaluation']['repeats'], 5)
                runs.add(config['run_id'])
        self.assertEqual(len(runs), 16)
        self.assertEqual(4*4*2*300, 9600)

    def test_parameters_reject_hardware_extra_seed_and_bool_shards(self):
        p = dict(cell='fedfisher_diag_shallow', variant='control', shard_index=0, shard_count=1)
        for change in ({'seed':0}, {'device':'cpu'}, {'shard_index':False}, {'shard_count':16}, {'cell':'fedavg'}):
            with self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.PILOT_TASK, dict(p, **change))
        config = protocol.run_config(protocol.PILOT_TASK, p)
        config['dgp']['num_test'] = 1
        self.assertEqual(protocol.run_config(protocol.PILOT_TASK, p)['dgp']['num_test'], 10000)

    def test_data_wrapper_matches_original_and_preserves_global_rows(self):
        p = dict(cell='fedfisher_diag_shallow', variant='control', shard_index=0, shard_count=1)
        config = protocol.run_config(protocol.PILOT_TASK, p)
        config['dgp'].update(num_sites=2, covariate_dim=3, site_min_samples=10,
                             site_max_samples=14, num_test=17)
        for variant in protocol.VARIANTS:
            config['variant'] = variant
            for split in protocol.SPLITS:
                sites, global_train, test, labels = generate_datasets(config, 0, split)
                expected, old_test, old_labels = old_generate(SimpleNamespace(
                    **config['dgp'], seed=0, split=split, variant=variant))
                self.assertEqual([dataset_fingerprint(s) for s in sites],
                                 [dataset_fingerprint(s) for s in expected])
                self.assertEqual(dataset_fingerprint(test), dataset_fingerprint(old_test))
                self.assertEqual(labels, old_labels)
                self.assertEqual(len(global_train), sum(map(len, sites)))

    def test_pair_gate_rejects_data_rng_and_evaluation_changes(self):
        from effect_modifier_experiment.fedfisher_timing.acceptance import validate_pairs
        records = paired_records()
        self.assertEqual(validate_pairs(records)['diag_kfac_local_fit_pairs'], 16)
        mutations = [
            lambda r:r.pop(next(iter(r))),
            lambda r:r['fedfisher_diag_shallow','local_permutation','iid']['fingerprints'].update(model='changed'),
            lambda r:r['fedfisher_kfac_deep','control','noniid']['fit'].update(local_model_fingerprints=['changed']),
            lambda r:r['fedfisher_kfac_deep','control','iid']['evaluation'].update(permutation_sha256='changed'),
        ]
        for mutate in mutations:
            changed = deepcopy(records)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_pairs(changed)

    def test_central_exit_and_identity_are_required(self):
        from effect_modifier_experiment.fedfisher_timing import acceptance
        scratch = Path('/scr/del6500/FedFisher/fedfisher-timing-validation')
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            request = dict(job_id='fixture', parameters=dict(cell='fedfisher_diag_shallow',
                variant='control', shard_index=0, shard_count=1))
            for name in ('stdout', 'stderr'):
                (root/('fixture.'+name)).write_text('')
            job = dict(schema_version=2, project='FedFisher', task=protocol.PILOT_TASK,
                job_id='fixture', parameters=request['parameters'], state='completed', exit_code=0,
                stdout_log=str(root/'fixture.stdout'), stderr_log=str(root/'fixture.stderr'))
            with mock.patch.object(acceptance,'CENTRAL_JOBS',root), mock.patch.object(acceptance,'CENTRAL_LOGS',root):
                (root/'fixture.json').write_text(json.dumps(job))
                self.assertEqual(acceptance._completed_job(request)[0], job)
                for change in ({'state':'failed'}, {'exit_code':1}, {'exit_code':False}, {'task':'other'}):
                    (root/'fixture.json').write_text(json.dumps(dict(job, **change)))
                    with self.assertRaises(ValueError):
                        acceptance._completed_job(request)

    def test_resume_accepts_earlier_producer_and_binds_its_exact_attempt(self):
        from effect_modifier_experiment.fedfisher_timing import acceptance
        scratch = Path('/scr/del6500/FedFisher/fedfisher-timing-validation')
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            job_id = 'fedfisher-timing-retry-fixture'
            allocation = dict(cpu_cores=1,memory_mib=8192,gpu_count=0,gpu_memory_mib=0,
                              gpu_utilization_pct=0,exclusive_gpu=False)
            started = dict(job_id=job_id,execution_profile='cpu-1core',allocation=allocation,
                           attempt=1,estimated_runtime_seconds=10.0)
            job = dict(job_id=job_id,execution_profile='cpu-1core',allocation=allocation,
                       estimated_runtime_seconds=50.0)
            for attempt in (1,2):
                for kind in ('stdout','stderr'):
                    path = root/('%s.attempt-%03d.%s.log' % (job_id,attempt,kind))
                    path.write_text('producer output\n' if kind == 'stdout' else '')
                    if attempt == 2:
                        job[kind+'_log'] = str(path)
            retry = dict(schema_version=2,job_id=job_id,attempts_started=2,max_attempts=3,
                         next_attempt_at=None,last_failure_reason='cpu_interrupted',updated_at='fixture')
            (root/(job_id+'.json')).write_text(json.dumps(retry))
            audit = []
            for attempt in (1,2):
                audit.extend([dict(schema_version=2,event='job_attempt_started',job_id=job_id,
                                   attempt=attempt,max_attempts=3,timestamp='fixture'),
                              dict(schema_version=2,event='job_transition',job_id=job_id,
                                   project='FedFisher',source='pending',destination='running',timestamp='fixture')])
            audit_path = root/'events.jsonl'
            audit_path.write_text(''.join(json.dumps(record)+'\n' for record in audit))
            with mock.patch.object(acceptance,'CENTRAL_LOGS',root), \
                    mock.patch.object(acceptance,'CENTRAL_RETRIES',root), \
                    mock.patch.object(acceptance,'CENTRAL_AUDIT',audit_path):
                proof = acceptance._producer_evidence(dict(job_id=job_id),job,started)
                self.assertTrue(proof['resumed_from_earlier_attempt'])
                self.assertEqual(proof['producer_attempt'],1)
                self.assertEqual(proof['final_completed_attempt'],2)
                self.assertEqual(proof['execution_receipt']['estimated_runtime_seconds'],10.0)
                self.assertTrue(proof['logs']['stdout_log']['path'].endswith('attempt-001.stdout.log'))
                self.assertEqual(proof['audit']['attempt_started']['record']['attempt'],1)
                acceptance._recheck_producer_evidence(proof)
                with audit_path.open('a') as stream:
                    stream.write(json.dumps(dict(schema_version=2,event='job_attempt_started',
                        job_id='unrelated',attempt=1))+'\n')
                acceptance._recheck_producer_evidence(proof)
                self.assertEqual(acceptance._producer_evidence(dict(job_id=job_id),job,started),proof)
                for change in ({'attempt':3}, {'job_id':'fedfisher-timing-wrong-job'}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        acceptance._producer_evidence(dict(job_id=job_id),job,dict(started,**change))
                # A receipt from the final attempt must still match that exact
                # central snapshot, including its runtime estimate.
                with self.assertRaisesRegex(ValueError,'its final central attempt'):
                    acceptance._producer_evidence(dict(job_id=job_id),job,dict(started,attempt=2))
                final = acceptance._producer_evidence(dict(job_id=job_id),job,
                    dict(started,attempt=2,estimated_runtime_seconds=50.0))
                self.assertFalse(final['resumed_from_earlier_attempt'])
                # Producer logs alone are insufficient: its exact central
                # start/running evidence must exist and remain unchanged.
                changed = deepcopy(audit)
                changed[0]['attempt'] = 7
                audit_path.write_text(''.join(json.dumps(record)+'\n' for record in changed))
                with self.assertRaisesRegex(ValueError,'start/running audit evidence'):
                    acceptance._producer_evidence(dict(job_id=job_id),job,started)
                with self.assertRaisesRegex(ValueError,'audit line changed'):
                    acceptance._recheck_producer_evidence(proof)


if __name__ == '__main__':
    unittest.main()
