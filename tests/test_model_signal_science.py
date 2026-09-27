"""Bounded CPU checks for the two additive, single-variable experiments."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from data import _sample_synthetic_effect_modifier
from models import SYNTHETIC_MLP_HIDDEN_DIMS, get_model
from effect_modifier_experiment.causal_comparison import protocol as comparison
from effect_modifier_experiment.causal_comparison import runner as comparison_runner
from effect_modifier_experiment.causal_comparison.evaluation import model_arrays
from effect_modifier_experiment.causal_comparison.fedavg import FedAvgConfig, validate_fit_report
from effect_modifier_experiment.causal_comparison.measurement import MethodTimer
from effect_modifier_experiment.dslassocov_benchmark import (
    dataset_fingerprint, tensor_datasets_to_raw_sites, training_fingerprint,
)
from effect_modifier_experiment.fedfisher_timing import protocol as timing
from effect_modifier_experiment.fedfisher_timing.data import generate_datasets
from effect_modifier_experiment.fedfisher_timing.fit import fit_fedfisher
from effect_modifier_experiment.model_signal_ablation import protocol
from effect_modifier_experiment.run_dslassocov_effect_modifier import generate_datasets as comparison_data


def parameters(setting='small_mlp', cell='fedavg_small_e1_budget30', **changes):
    result = dict(setting=setting, cell=cell, variant='control', shard_index=0, shard_count=1)
    result.update(changes)
    return result


def tiny_config(setting='small_mlp', cell='fedavg_small_e1_budget30', variant='control'):
    config = protocol.run_config(protocol.PILOT_TASK, parameters(setting, cell, variant=variant))
    config['dgp'].update(covariate_dim=3, num_sites=2, num_test=16,
                         site_min_samples=12, site_max_samples=16)
    config['evaluation'].update(repeats=1, batch_size=8)
    if config['method']['family'] == 'fedavg':
        config['method']['optimizer'] = asdict(FedAvgConfig(
            covariate_dim=3, local_epochs=1, batch_size=8, min_rounds=1, max_rounds=2,
            stopping_mode='fixed_total_epochs', total_local_epochs=2))
    elif config['method']['family'] == 'legacy_fedfisher':
        config['method']['optimizer'].update(local_epochs=1, batch_size=8)
    return config


class ModelSignalProtocolTests(unittest.TestCase):
    def test_exact_requested_coverage_and_independent_namespaces(self):
        neural = {
            'fedavg_%s_e%d_%s' % (depth, epochs, mode)
            for depth in ('small', 'shallow', 'deep')
            for epochs in (1, 5, 10, 30) for mode in ('patience', 'budget30')
        } | {'fedfisher_%s_%s' % (kind, depth)
             for kind in ('diag', 'kfac') for depth in ('small', 'shallow', 'deep')}
        expected = {'small_mlp': {cell for cell in neural if '_small' in cell},
                    'weak_signal_1': neural | {'dslassocov'},
                    'weak_signal_05': neural | {'dslassocov'}}
        self.assertEqual(set(protocol.SETTINGS), set(expected))
        self.assertEqual(set(protocol.CELLS), neural | {'dslassocov'})
        self.assertEqual(sum(len(cells) for cells in expected.values()), 72)
        names = set()
        for setting, cells in expected.items():
            self.assertEqual(set(protocol.SETTINGS[setting]['cells']), cells)
            for cell in cells:
                for variant in comparison.VARIANTS:
                    p = parameters(setting, cell, variant=variant)
                    for task, count in ((protocol.PILOT_TASK, 1), (protocol.FORMAL_TASK, 8)):
                        config = protocol.run_config(task, dict(p, shard_count=count))
                        self.assertEqual(config['setting'], setting)
                        self.assertEqual(config['seeds'], [0, 0 if count == 1 else 299])
                        self.assertEqual(config['splits'], ['iid', 'noniid'])
                        self.assertNotIn(config['run_id'], names)
                        self.assertFalse(config['run_id'].startswith(('comparison_', 'fedfisher_timing_')))
                        names.add(config['run_id'])
        self.assertEqual(len(names), 72 * 4 * 2)
        # Merely making a third factory model available must not enlarge old jobs.
        self.assertEqual(len(comparison.CELLS), 24)
        self.assertEqual(len(timing.CELLS), 4)
        self.assertFalse(any('_small' in cell for cell in (*comparison.CELLS, *timing.CELLS)))

    def test_only_model_or_signal_changes_from_original_scientific_recipes(self):
        for setting, strength in (('small_mlp', 2.), ('weak_signal_1', 1.), ('weak_signal_05', .5)):
            for cell in protocol.SETTINGS[setting]['cells']:
                with self.subTest(setting=setting, cell=cell):
                    config = protocol.run_config(protocol.PILOT_TASK, parameters(setting, cell))
                    self.assertEqual(config['dgp'], dict(comparison.DGP, signal_strength=strength))
                    self.assertEqual(config['evaluation'], comparison.EVALUATION)
                    original_cell = cell.replace('_small', '_shallow')
                    source = timing if cell.startswith('fedfisher_') else comparison
                    original = source.method_config(original_cell)
                    if '_small' in cell:
                        original['model'] = 'SyntheticMLPSmall'
                    self.assertEqual(config['method'], original)
                    self.assertEqual(config['dgp']['main_effect_strength'], 1.)
                    self.assertEqual(config['dgp']['confounding_strength'], 1.)
        # The fixed-budget E30 arm still contains exactly one communication round.
        method = protocol.method_config('fedavg_small_e30_budget30')
        self.assertEqual(method['optimizer']['total_local_epochs'] // method['optimizer']['local_epochs'], 1)

    def test_returned_configurations_cannot_mutate_other_settings_or_old_defaults(self):
        before = deepcopy((comparison.DGP, comparison.EVALUATION, timing.OPTIMIZER))
        config = protocol.run_config(protocol.PILOT_TASK, parameters())
        original = deepcopy(config)
        config['dgp']['signal_strength'] = 99
        config['evaluation']['modes'].append('foreign')
        config['method']['optimizer']['learning_rate'] = 99
        self.assertEqual(protocol.run_config(protocol.PILOT_TASK, parameters()), original)
        self.assertEqual((comparison.DGP, comparison.EVALUATION, timing.OPTIMIZER), before)

    def test_cross_setting_cells_and_arbitrary_science_are_rejected(self):
        for change in ({'cell': 'dslassocov'}, {'cell': 'fedfisher_diag_deep'},
                       {'setting': 'weak_signal_2'}, {'signal_strength': .2},
                       {'model': 'SyntheticMLPSmall'}, {'seed': 3}, {'cpu_cores': 2},
                       {'setting': []}, {'cell': []}, {'variant': 'unreviewed'},
                       {'shard_index': True}, {'shard_count': True}, {'shard_index': 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.PILOT_TASK, parameters(**change))
        for old_task in (*comparison.TASKS, *timing.TASKS):
            with self.assertRaises(ValueError):
                protocol.validate_parameters(old_task, parameters())
        for count in (8, 16):
            shards = [protocol.seed_sequence(protocol.FORMAL_TASK, parameters(
                'weak_signal_05', 'dslassocov', shard_index=index, shard_count=count))
                for index in range(count)]
            self.assertEqual(sorted(seed for shard in shards for seed in shard), list(range(300)))
            self.assertTrue(all(all(seed % count == index for seed in shard)
                                for index, shard in enumerate(shards)))


class ModelSignalNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_factory_has_one_hidden_layer_eight_neurons_and_1634_parameters(self):
        self.assertEqual(SYNTHETIC_MLP_HIDDEN_DIMS['SyntheticMLPSmall'], (8,))
        self.assertEqual(SYNTHETIC_MLP_HIDDEN_DIMS['SyntheticMLP'], (64, 32))
        self.assertEqual(SYNTHETIC_MLP_HIDDEN_DIMS['SyntheticMLPDeep'], (256, 128, 64, 32))
        model = get_model('SyntheticMLPSmall', 2, synthetic_dim=201)
        self.assertEqual([type(layer) for layer in model.net], [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear])
        self.assertEqual({name: tuple(value.shape) for name, value in model.named_parameters()}, {
            'net.0.weight': (8, 201), 'net.0.bias': (8,), 'net.2.weight': (2, 8), 'net.2.bias': (2,)})
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1634)
        self.assertEqual(tuple(model(torch.zeros(3, 201)).shape), (3, 2))

    def test_real_small_fedavg_rebroadcasts_the_aggregate_and_preserves_rng(self):
        config = tiny_config()
        with redirect_stdout(io.StringIO()):
            sites, _, _, _ = generate_datasets(config, 9, 'iid')
        raw = tensor_datasets_to_raw_sites(sites, 3)
        before = torch.random.get_rng_state().clone()
        model, report = comparison_runner.fit_method(raw, sites, config['method'], 9, MethodTimer(config['cell']))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        arrays = model_arrays(model)
        self.assertEqual(arrays['net.0.weight'].shape, (8, 7))
        validate_fit_report(report, config['method']['optimizer'], 'SyntheticMLPSmall', 9, arrays)
        self.assertEqual(report['training_rounds_executed'], 2)
        self.assertEqual(report['round_history'][1]['initial_model_fingerprint'],
                         report['round_history'][0]['aggregate_model_fingerprint'])
        self.assertTrue(report['completion_eligible'])
        self.assertEqual(report['training_examples_seen_per_site'], [2 * len(site) for site in sites])

    def test_real_small_diag_and_kfac_share_local_fits_without_mocking_the_model_factory(self):
        config = tiny_config(cell='fedfisher_diag_small')
        with redirect_stdout(io.StringIO()):
            sites, global_training, _, _ = generate_datasets(config, 9, 'iid')
        records = {}
        for kind in ('diag', 'kfac'):
            method = deepcopy(config['method'])
            method['algorithm'] = 'fedfisher_' + kind
            timer = MethodTimer(method['algorithm'])
            before = torch.random.get_rng_state().clone()
            with redirect_stdout(io.StringIO()):
                model, report = fit_fedfisher(sites, method, 9, timer, global_training=global_training)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(model.net[0].out_features, 8)
            self.assertTrue(all(np.isfinite(array).all() for array in model_arrays(model).values()))
            self.assertTrue(report['completion_eligible'])
            self.assertEqual(report['server_iterations'], 2000)
            self.assertEqual(report['validation_checks'], 20)
            self.assertEqual(set(timer.to_dict()['stages']['train']['components']), {'sgd', 'fisher_' + kind})
            records[kind] = report
        for field in ('initial_model_fingerprint', 'local_model_fingerprints', 'validation', 'training_fingerprint'):
            self.assertEqual(records['diag'][field], records['kfac'][field])

    def test_both_existing_data_paths_receive_the_signal_and_preserve_four_variant_pairs(self):
        for setting in ('small_mlp', 'weak_signal_1', 'weak_signal_05'):
            for split in ('iid', 'noniid'):
                observed = {}
                for variant in comparison.VARIANTS:
                    config = tiny_config(setting=setting, variant=variant)
                    with redirect_stdout(io.StringIO()), mock.patch(
                            'data._sample_synthetic_effect_modifier', wraps=_sample_synthetic_effect_modifier) as draws:
                        sites, _, test, labels = generate_datasets(config, 8, split)
                        other_sites, other_test, other_labels = comparison_data(SimpleNamespace(
                            **config['dgp'], seed=8, split=split, variant=variant))
                    self.assertTrue(draws.called)
                    for call in draws.call_args_list:
                        self.assertEqual(call.args[3:6], (config['dgp']['signal_strength'], 1., 1.))
                    fingerprint = training_fingerprint(tensor_datasets_to_raw_sites(sites, 3))
                    self.assertEqual(fingerprint, training_fingerprint(tensor_datasets_to_raw_sites(other_sites, 3)))
                    self.assertEqual(dataset_fingerprint(test), dataset_fingerprint(other_test))
                    self.assertEqual(labels, other_labels)
                    observed[variant] = (fingerprint, dataset_fingerprint(test), [len(site) for site in sites], labels)
                self.assertEqual(len({row[1] for row in observed.values()}), 1)
                self.assertEqual(observed['control'], observed['local_permutation'])
                self.assertEqual(observed['site_gaussian_shift'], observed['site_gaussian_local_permutation'])
                self.assertNotEqual(observed['control'][0], observed['site_gaussian_shift'][0])
                self.assertEqual(observed['control'][2:], observed['site_gaussian_shift'][2:])


if __name__ == '__main__':
    unittest.main()
