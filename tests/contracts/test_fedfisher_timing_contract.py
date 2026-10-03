"""Tiny immutable-result and timing semantics fixtures; no scientific pilot."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from models import SYNTHETIC_MLP_HIDDEN_DIMS
from effect_modifier_experiment.causal_comparison.evaluation import model_hash, signal_recovery_with_ties
from effect_modifier_experiment.causal_comparison.measurement import MethodTimer
from effect_modifier_experiment.dslassocov_run_contract import canonical_text
from effect_modifier_experiment.fedfisher_timing import artifacts, protocol, validation


def tiny_configuration(kind='diag', variant='control'):
    config = protocol.run_config(protocol.PILOT_TASK, dict(cell='fedfisher_'+kind+'_shallow',
        variant=variant, shard_index=0, shard_count=1))
    config['dgp'].update(covariate_dim=3, num_sites=2, num_test=8, site_min_samples=6, site_max_samples=8)
    config['method']['optimizer'].update(local_epochs=2, batch_size=4)
    return config


def provenance():
    values = dict(git_commit='1'*40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                  source_tree_sha256='2'*64)
    values['provenance_sha256'] = hashlib.sha256(canonical_text(values, artifacts.PROVENANCE_KEYS).encode()).hexdigest()
    return values


def execution():
    return dict(job_id='fedfisher-timing-contract-fixture', execution_profile='cpu-1core', attempt=1,
        estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0,
        gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def observations(config, identity):
    method = config['method']
    options = method['optimizer']
    widths = (7, *SYNTHETIC_MLP_HIDDEN_DIMS[method['model']], 2)
    arrays = {}
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        arrays['net.%d.weight' % (2*index)] = np.zeros((right, left), dtype=np.float32)
        arrays['net.%d.bias' % (2*index)] = np.zeros(right, dtype=np.float32)
    sizes, reports = [6, 8], []
    kind = method['algorithm'].removeprefix('fedfisher_')
    for index, size in enumerate(sizes):
        reports.append(dict(site_idx=index, num_examples=size, weight=size/14, epochs=2, updates=4,
            sample_visits=size*2, model_fingerprint='3'*64, fisher_fingerprint='4'*64, training_fingerprint='5'*64))
    indices = np.random.RandomState(3).choice(14, 14, replace=False)
    fit = dict(family='legacy_fedfisher', algorithm=method['algorithm'], model=method['model'], seed=0,
        completion_eligible=True, termination_reason='fixed_recipe_completed', convergence_claim=False,
        device='cpu', rng_policy=options['rng_policy'], standalone_method=True, fisher_representation=kind,
        historical_gpu_bitwise_reproduction=False, local_epochs=2,
        parameter_count=sum(value.size for value in arrays.values()), total_local_updates=8, total_sample_visits=28,
        training_communication_rounds=1, server_iterations=2000, validation_checks=20,
        initial_model_fingerprint='6'*64, model_fingerprint=model_hash(arrays), training_fingerprint='7'*64,
        site_reports=reports, local_model_fingerprints=[item['model_fingerprint'] for item in reports],
        validation=dict(scope='pooled_training_subset_also_used_in_local_fits', independent_test_access=False,
            pooled_order='original_generator_global_row_order', selection_seed=3, num_examples=14,
            indices_sha256=hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest(),
            dataset_sha256='8'*64, selection_rule='highest_validation_accuracy_strict_improvement_first_tie',
            checked_iteration_indices=list(range(0, 2000, 100))),
        timing_contract=dict(training_includes_all_selection_and_refits=True, fit_stages=['train', 'aggregation'],
            train='initialization, all local SGD and selected Fisher representation only',
            aggregation='original quantization/compression and all server iterations/validation selection',
            tuning='included_in_aggregation', refit='none_in_original_recipe',
            evaluation='independent_post_fit_stage_owned_by_experiment'),
        communication=dict(execution_model='single_process_simulation', actual_network_measured=False,
            accounted=False, training_communication_rounds=1,
            reason='preserved aggregation has no serialized upload boundary; KFAC compression accesses training rows'))
    timer = MethodTimer(config['cell'])
    for stage in ('data_preparation', 'train', 'aggregation', 'evaluation'):
        with timer.measure(stage):
            if stage == 'train':
                for _ in sizes:
                    with timer.measure('train', component='sgd'):
                        pass
                    with timer.measure('train', component='fisher_'+kind):
                        pass
    importance = [0., 0., 0.]
    metrics = {mode+'/'+scale: dict(importance=importance, recovery=signal_recovery_with_ties(
        importance, 2, 'tau_intervention_'+scale.removeprefix('tau_')))
        for mode in ('permute', 'zero') for scale in ('tau_prob', 'tau_logit')}
    local = 'local_permutation' in config['variant']
    result = dict(status='complete', identity=identity, fit=fit, timing=timer.to_dict(),
        memory=dict(process_peak_rss_kib=1024, scope='Linux current-process lifetime high-water RSS; not cgroup peak or reservation'),
        evaluation=dict(primary='permute/tau_prob', effect_only=False, evaluation_network_measured=False,
            metrics=metrics, raw_x_sha256='9'*64, permutation_sha256='a'*64,
            shard_sizes=[3, 5] if local else [8], shard_indices_sha256=['b'*64]*(2 if local else 1)),
        site_sizes=sizes, label_counts={'0': {'0': 3, '1': 3}, '1': {'0': 4, '1': 4}},
        fingerprints=dict(training='7'*64, test='c'*64, model=model_hash(arrays)))
    return result, arrays


class TimingSemanticTests(unittest.TestCase):
    def setUp(self):
        self.config = tiny_configuration()
        self.identity = dict(seed=0, fixture='semantic')
        self.result, self.arrays = observations(self.config, self.identity)

    def test_both_selected_fishers_have_valid_exclusive_stage_timing_and_ties(self):
        for kind in ('diag', 'kfac'):
            config = tiny_configuration(kind, variant='local_permutation')
            result, arrays = observations(config, self.identity)
            validation.validate_result(result, arrays, self.identity, config)
            self.assertEqual(set(result['timing']['stages']['train']['components']), {'sgd', 'fisher_'+kind})
            self.assertEqual(result['evaluation']['metrics']['permute/tau_prob']['recovery']['expected_topk_hits'], 4/3)

    def test_total_cannot_double_count_components_or_drop_aggregation_selection(self):
        cases = []
        doubled = deepcopy(self.result)
        doubled['timing']['measured_stage_sum']['wall_seconds'] += sum(
            row['wall_seconds'] for row in doubled['timing']['stages']['train']['components'].values())
        cases.append(doubled)
        no_aggregation = deepcopy(self.result)
        no_aggregation['timing']['stages']['aggregation'].update(measured=False, calls=0, wall_seconds=0, cpu_seconds=0)
        cases.append(no_aggregation)
        for result in cases:
            with self.assertRaises(ValueError):
                validation.validate_result(result, self.arrays, self.identity, self.config)

    def test_zero_durations_cannot_claim_measured_standalone_timing(self):
        result = deepcopy(self.result)
        for stage in result['timing']['stages'].values():
            stage.update(wall_seconds=0., cpu_seconds=0.)
            for component in stage['components'].values():
                component.update(wall_seconds=0., cpu_seconds=0.)
        result['timing']['measured_stage_sum'] = dict(wall_seconds=0., cpu_seconds=0.)
        with self.assertRaisesRegex(ValueError, 'no observed wall duration'):
            validation.validate_result(result, self.arrays, self.identity, self.config)
        result = deepcopy(self.result)
        result['timing']['stages']['train']['components']['fisher_diag']['wall_seconds'] = 0.
        with self.assertRaisesRegex(ValueError, 'no observed wall duration'):
            validation.validate_result(result, self.arrays, self.identity, self.config)

    def test_tiny_real_runner_result_passes_complete_semantic_contract(self):
        import torch
        from effect_modifier_experiment.fedfisher_timing import runner
        torch.set_num_threads(1)
        config = tiny_configuration()
        config['dgp'].update(site_min_samples=10, site_max_samples=10)
        config['method']['optimizer']['local_epochs'] = 1
        config['evaluation'].update(repeats=1, batch_size=4)
        with redirect_stdout(io.StringIO()):
            result, arrays = runner.compute_cell(config, 0, 'iid')
        result['identity'] = self.identity
        validation.validate_result(result, arrays, self.identity, config)
        self.assertEqual(result['fit']['total_sample_visits'], 20)
        self.assertEqual(result['fit']['validation']['pooled_order'], 'original_generator_global_row_order')
        self.assertEqual(result['timing']['stages']['train']['components']['fisher_diag']['calls'], 2)

    def test_missing_extra_or_failed_fisher_observation_is_not_standalone_timing(self):
        for change in ('missing', 'extra', 'wrong_count', 'failed', 'exceeds_stage'):
            with self.subTest(change=change):
                result = deepcopy(self.result)
                stage = result['timing']['stages']['train']
                components = stage['components']
                if change == 'missing':
                    del components['fisher_diag']
                elif change == 'extra':
                    components['fisher_kfac'] = deepcopy(components['fisher_diag'])
                elif change == 'wrong_count':
                    components['fisher_diag']['calls'] = 1
                elif change == 'failed':
                    components['fisher_diag']['failed_calls'] = 1
                else:
                    components['fisher_diag']['wall_seconds'] = stage['wall_seconds']+1
                with self.assertRaises(ValueError):
                    validation.validate_result(result, self.arrays, self.identity, self.config)

    def test_fit_accounting_selection_and_training_identity_are_checked(self):
        for path, replacement in (
                (('fit', 'total_local_updates'), 7), (('fit', 'total_sample_visits'), 27),
                (('fit', 'training_communication_rounds'), 2000), (('fit', 'convergence_claim'), True),
                (('fit', 'validation', 'pooled_order'), 'site_index_then_local_row_fixture_fallback'),
                (('fit', 'validation', 'indices_sha256'), 'd'*64),
                (('fit', 'validation', 'independent_test_access'), True),
                (('fit', 'model_fingerprint'), 'e'*64), (('fit', 'training_fingerprint'), 'f'*64),
                (('fit', 'site_reports', 0, 'weight'), .5)):
            with self.subTest(path=path):
                result = deepcopy(self.result)
                target = result
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaises(ValueError):
                    validation.validate_result(result, self.arrays, self.identity, self.config)

    def test_model_shape_dtype_and_recovery_semantics_cannot_be_rehashed_away(self):
        for change in ('array', 'dtype', 'shape', 'tie_metric', 'test_size', 'labels', 'memory', 'clock'):
            with self.subTest(change=change):
                result, arrays = deepcopy(self.result), deepcopy(self.arrays)
                name = next(iter(arrays))
                if change == 'array':
                    arrays[name].flat[0] = 1
                elif change == 'dtype':
                    arrays[name] = arrays[name].astype(np.float64)
                elif change == 'shape':
                    arrays[name] = arrays[name][:-1]
                elif change == 'tie_metric':
                    result['evaluation']['metrics']['permute/tau_prob']['recovery']['expected_topk_hits'] = 2
                elif change == 'test_size':
                    result['evaluation']['shard_sizes'] = [7]
                elif change == 'labels':
                    result['label_counts']['0']['0'] = 4
                elif change == 'memory':
                    result['memory']['process_peak_rss_kib'] = 0
                else:
                    result['timing']['clock']['cpu'] = 'queue_duration'
                with self.assertRaises(ValueError):
                    validation.validate_result(result, arrays, self.identity, self.config)


class TimingArtifactTests(unittest.TestCase):
    def setUp(self):
        root = artifacts.safe_path(Path('/scr/del6500/FedFisher/fedfisher-timing-validation/contracts'))
        root.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=root)))
        self.config = tiny_configuration()
        self.provenance = provenance()
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=deepcopy(self.config)))
        self.current = self.stack.enter_context(mock.patch.object(artifacts, 'current_provenance', return_value=self.provenance))
        self.store = artifacts.ArtifactStore(self.config, self.provenance, self.root/self.config['run_id'])
        self.store.initialize()
        self.result, self.arrays = observations(self.config, self.store.identity(0, 'iid'))

    def complete(self):
        self.store.start(0, 'iid', execution())
        self.store.complete(0, 'iid', self.result, self.arrays)

    def test_completed_cell_revalidates_read_only_and_never_overwrites(self):
        self.assertEqual(self.store.status(0, 'iid'), 'missing')
        self.complete()
        paths = list(self.store.directory.rglob('*'))
        before = {str(path): path.read_bytes() for path in paths if path.is_file()}
        self.assertEqual(self.store.status(0, 'iid'), 'validated')
        self.assertEqual(self.store.status(0, 'noniid'), 'missing')
        after = {str(path): path.read_bytes() for path in paths if path.is_file()}
        self.assertEqual(before, after)
        with self.assertRaises(FileExistsError):
            self.store.start(0, 'iid', execution())
        with self.assertRaisesRegex(ValueError, 'overwritten'):
            self.store.complete(0, 'iid', self.result, self.arrays)
        with self.assertRaisesRegex(ValueError, 'marked failed'):
            self.store.fail(0, 'iid', RuntimeError('late failure'))

    def test_partial_and_failed_cells_remain_unaccepted(self):
        self.store.start(0, 'iid', execution())
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status(0, 'iid')
        bad = deepcopy(self.result)
        bad['timing']['stages']['train']['failed_calls'] = 1
        with self.assertRaises(ValueError):
            self.store.complete(0, 'iid', bad, self.arrays)
        self.store.fail(0, 'iid', RuntimeError('fixture failure'))
        self.assertEqual({path.name for path in self.store.cell_dir(0, 'iid').iterdir()}, {'started.json', 'failure.json'})
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status(0, 'iid')

    def test_rehashed_result_or_execution_tampering_still_fails_semantic_validation(self):
        self.complete()
        path = self.store.cell_dir(0, 'iid')
        original_result = (path/'result.json').read_bytes()
        result = json.loads(original_result)
        result['timing']['stages']['train']['components']['fisher_kfac'] = deepcopy(
            result['timing']['stages']['train']['components']['fisher_diag'])
        (path/'result.json').write_text(json.dumps(result))
        self.rehash(path, 'result.json')
        with self.assertRaisesRegex(ValueError, 'extra work'):
            self.store.status(0, 'iid')
        (path/'result.json').write_bytes(original_result)
        self.rehash(path, 'result.json')
        started = json.loads((path/'started.json').read_text())
        started['execution']['attempt'] = 0
        (path/'started.json').write_text(json.dumps(started))
        self.rehash(path, 'started.json')
        with self.assertRaisesRegex(ValueError, 'profile or attempt'):
            self.store.status(0, 'iid')

    @staticmethod
    def rehash(path, name):
        marker = json.loads((path/'done.json').read_text())
        marker['artifacts'][name] = hashlib.sha256((path/name).read_bytes()).hexdigest()
        (path/'done.json').write_text(json.dumps(marker))

    def test_source_drift_blocks_reads_and_cannot_publish_completion_marker(self):
        self.store.start(0, 'iid', execution())
        calls = []

        def changing_validation(*args):
            validation.validate_result(*args)
            calls.append(1)
            if len(calls) == 2:
                self.current.return_value = dict(self.provenance, source_tree_sha256='9'*64)

        with mock.patch.object(artifacts, 'validate_result', side_effect=changing_validation):
            with self.assertRaisesRegex(ValueError, 'current source provenance'):
                self.store.complete(0, 'iid', self.result, self.arrays)
        self.assertFalse((self.store.cell_dir(0, 'iid')/'done.json').exists())
        with self.assertRaisesRegex(ValueError, 'current source provenance'):
            self.store.status(0, 'iid')

    def test_unreviewed_configuration_symlinks_and_foreign_identity_fail_closed(self):
        changed = deepcopy(self.config)
        changed['method']['optimizer']['local_epochs'] = 3
        with self.assertRaisesRegex(ValueError, 'reviewed fixed recipe'):
            artifacts.ArtifactStore(changed, self.provenance, self.root/changed['run_id'])
        for seed, split in ((True, 'iid'), (1, 'iid'), (0, 'outside')):
            with self.assertRaisesRegex(ValueError, 'scientific unit'):
                self.store.identity(seed, split)
        link = self.root/'linked'
        link.symlink_to(self.store.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            artifacts.ArtifactStore(self.config, self.provenance, link/self.config['run_id'])
        self.store.start(0, 'iid', execution())
        changed = deepcopy(self.result)
        changed['identity']['split'] = 'noniid'
        with self.assertRaisesRegex(ValueError, 'identity/status'):
            self.store.complete(0, 'iid', changed, self.arrays)


if __name__ == '__main__':
    unittest.main()
