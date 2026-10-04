"""Bounded pooled result/lifecycle fixtures; saved validation performs no inference."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from effect_modifier_experiment.infrastructure.provenance import PROVENANCE_KEYS, canonical_text
from effect_modifier_experiment.survival import artifacts as original_artifacts
from effect_modifier_experiment.survival.data import SurvivalRows, SurvivalSite
from effect_modifier_experiment.survival.fisher import FisherFit
from effect_modifier_experiment.survival.links import survival_mean
from effect_modifier_experiment.survival_pooled import artifacts, inputs, protocol
from effect_modifier_experiment.survival_pooled.evaluation import compare_predictions, summarize_arrays
from effect_modifier_experiment.survival_pooled.methods import fit_pooled

SCRATCH = Path('/scr/del6500/FedFisher/survival-pooled-validation/contracts')


def provenance():
    value = dict(git_commit='1' * 40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                 source_tree_sha256='2' * 64)
    value['provenance_sha256'] = hashlib.sha256(canonical_text(value, PROVENANCE_KEYS).encode()).hexdigest()
    return value


def execution():
    return dict(job_id='fedfisher-survival-pooled-contract-fixture', execution_profile='cpu-1core',
        attempt=1, estimated_runtime_seconds=20., allocation=dict(cpu_cores=1, memory_mib=8192,
        gpu_count=0, gpu_memory_mib=0, gpu_utilization_pct=0, exclusive_gpu=False))


def tiny_example():
    """Two bounded fitting fixtures; prediction arrays are specified analytically."""
    config = protocol.run_config(protocol.PILOT_TASK, dict(family='shallow', scenario='iid'))
    config['dgp'].update(num_sites=1, site_min_samples=8, site_max_samples=8,
                         covariate_dim=2, num_test=4)
    config['method']['optimizer'].update(epochs=1, batch_size=4)
    rows = SurvivalRows(np.array([[-1., 0.], [.1, .3], [.2, -.1], [.4, .5],
                                 [.3, .8], [-.2, -.9], [.7, -.1], [-.5, .4]]),
                        np.tile([0., 1.], 4), np.tile([1., 2., 4., 5.], 2),
                        np.ones(8), np.arange(8))
    truth = dict(propensity=np.array([.3, .4, .6, .7]), eta0=np.array([-1., .2, .7, 3.]),
                 delta=np.array([.23, -.31, 1.27, -.4]), modifier_mask=np.array([True, True]))
    truth['eta1'] = truth['eta0'] + truth['delta']
    truth.update(survival0=survival_mean(truth['eta0']), survival1=survival_mean(truth['eta1']))
    probability_truth = {key: truth[key] for key in ('eta0', 'eta1', 'survival0', 'survival1')}
    treatment = np.array([0., 1., 0., 1.])
    arrays = {'truth__' + key: value.copy() for key, value in truth.items()}
    arrays.update(factual_a=treatment, plot_x12=np.zeros((4, 2)))
    fits = {}
    with mock.patch.object(FisherFit, 'predict_eta', side_effect=AssertionError('no inference')):
        for scope in config['pseudo_scopes']:
            _, report, fitted = fit_pooled([SurvivalSite(0, rows)], 3., 0, 'shallow', scope,
                                           config['method'], config['public_initial_eta'])
            fits[scope] = report
            arrays.update({'pooled_' + scope + '__fit__' + key: value.copy()
                           for key, value in fitted.items()})
    # Distinct manual predictors make the saved comparison arithmetic testable
    # without loading or applying any fitted or historical network.
    predictions = {}
    tags = [*protocol.historical_tags('shallow'),
            *('pooled_' + scope for scope in config['pseudo_scopes'])]
    for index, tag in enumerate(tags):
        eta0, eta1 = truth['eta0'] + .03 * index, truth['eta1'] - .02 * index
        predictions[tag] = dict(eta0=eta0, eta1=eta1, s0=survival_mean(eta0), s1=survival_mean(eta1))
        arrays.update({tag + '__prediction__' + key: value.copy()
                       for key, value in predictions[tag].items()})
    fingerprints = dict(training='4' * 64, evaluation='5' * 64,
                        evaluation_truth=original_artifacts.array_hash(truth))
    result = dict(status='complete', fit=fits,
        prediction={tag: summarize_arrays(prediction, probability_truth, treatment)
                    for tag, prediction in predictions.items()},
        comparisons={tag: {scope: compare_predictions(predictions[tag], predictions['pooled_' + scope])
                           for scope in config['pseudo_scopes']}
                     for tag in protocol.historical_tags('shallow')},
        historical={tag: dict(identity=dict(seed=0, scenario='iid', cell=tag + '_shallow'))
                    for tag in protocol.historical_tags('shallow')},
        historical_inputs=inputs.historical_binding(), site_sizes=[8], fingerprints=fingerprints,
        timing=dict(stages={stage: dict(wall_seconds=100., cpu_seconds=100.)
                           for stage in config['timing']['stages']},
                    network_transport_measured=False, fit_includes_all_selection_and_refits=True),
        memory=dict(process_peak_rss_kib=1024,
                    scope='Linux process-lifetime high-water RSS, not allocation or cgroup peak'),
        prediction_arrays_sha256=original_artifacts.array_hash(arrays))
    return config, result, arrays


class SurvivalPooledContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.example = tiny_example()

    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.config, self.result, self.arrays = deepcopy(self.example)
        self.frozen_fingerprints = deepcopy(self.result['fingerprints'])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(original_artifacts, 'current_provenance', return_value=provenance()))
        # Only this explicitly bounded test recipe bypasses production defaults.
        self.stack.enter_context(mock.patch.object(protocol, 'run_config', return_value=deepcopy(self.config)))
        self.selected_check = self.stack.enter_context(mock.patch.object(inputs, 'validate_selected_binding', return_value=True))
        self.stack.enter_context(mock.patch.object(inputs, 'selected_fingerprints',
            side_effect=lambda receipt: deepcopy(self.frozen_fingerprints)))
        self.stack.enter_context(mock.patch.object(FisherFit, 'predict_eta', side_effect=AssertionError('no inference')))
        self.directory = Path(self.temporary.name) / self.config['run_id']
        self.store = artifacts.ArtifactStore(self.config, provenance(), directory=self.directory)
        self.result['identity'] = self.store.identity(0)

    def validate(self):
        return artifacts.validate_result(self.result, self.arrays, self.store.identity(0), self.config)

    def rehash_arrays(self):
        self.result['prediction_arrays_sha256'] = original_artifacts.array_hash(self.arrays)

    def publish(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.store.complete(self.result, self.arrays, 0)

    def test_valid_saved_arrays_and_model_endpoints_require_no_inference(self):
        self.assertTrue(self.validate())
        self.assertEqual(self.selected_check.call_count, 2)

    def test_all_seven_oracle_fields_bind_original_dtype_shape_and_bytes(self):
        truth = {key: self.arrays['truth__' + key] for key in
                 ('propensity', 'eta0', 'eta1', 'survival0', 'survival1', 'delta', 'modifier_mask')}
        self.assertEqual(original_artifacts.array_hash(truth), self.result['fingerprints']['evaluation_truth'])
        self.assertEqual(truth['modifier_mask'].dtype, np.bool_)
        self.arrays['truth__propensity'][0] += .01
        self.rehash_arrays()
        with self.assertRaisesRegex(ValueError, 'truth differs'):
            self.validate()

    def test_delta_checks_original_addition_without_float_subtraction_requirement(self):
        eta0, eta1, delta = (self.arrays['truth__' + key] for key in ('eta0', 'eta1', 'delta'))
        self.assertTrue(np.array_equal(eta1, eta0 + delta))
        self.assertFalse(np.array_equal(delta, eta1 - eta0))
        self.assertTrue(self.validate())

    def test_oracle_mask_and_probability_arrays_reject_wrong_dtypes(self):
        for key, dtype in [('truth__modifier_mask', np.float64), ('truth__eta0', np.float32),
                           ('truth__propensity', np.float32), ('factual_a', np.int64)]:
            with self.subTest(key=key):
                old = self.arrays[key]
                self.arrays[key] = old.astype(dtype)
                self.rehash_arrays()
                with self.assertRaises(ValueError):
                    self.validate()
                self.arrays[key] = old
                self.rehash_arrays()

    def test_saved_summary_and_pooled_gap_are_recomputed_from_arrays(self):
        self.result['prediction']['fedfisher_diag']['combined']['rmse'] += .1
        with self.assertRaisesRegex(ValueError, 'metrics differ'):
            self.validate()
        self.result = deepcopy(self.example[1])
        self.result['identity'] = self.store.identity(0)
        self.result['comparisons']['fedfisher_diag']['site_local_pv']['combined']['rmse'] += .1
        with self.assertRaisesRegex(ValueError, 'gap arithmetic'):
            self.validate()

    def test_historical_model_must_match_seed_scenario_family_and_anchor(self):
        self.result['historical']['fedfisher_diag']['identity']['seed'] = 49
        with self.assertRaisesRegex(ValueError, 'different scientific unit'):
            self.validate()
        self.result['historical']['fedfisher_diag']['identity']['seed'] = 0
        self.result['historical_inputs']['acceptance']['sha256'] = '9' * 64
        with self.assertRaisesRegex(ValueError, 'acceptance anchor'):
            self.validate()

    def test_historical_receipt_validation_failure_fails_closed(self):
        self.selected_check.side_effect = ValueError('mutated historical artifact')
        with self.assertRaisesRegex(ValueError, 'mutated historical'):
            self.validate()

    def test_training_fingerprint_cannot_be_relabelled_to_other_data(self):
        self.result['fingerprints']['training'] = '8' * 64
        with self.assertRaisesRegex(ValueError, 'differ from selected'):
            self.validate()

    def test_wrong_fit_seed_input_dimension_or_public_initialization_is_rejected(self):
        scope = self.config['pseudo_scopes'][0]
        for key, value in [('seed', 49), ('seed', True), ('input_dim', 7),
                           ('initial_eta', self.config['public_initial_eta'] + .1)]:
            with self.subTest(key=key, value=value):
                old = self.result['fit'][scope][key]
                self.result['fit'][scope][key] = value
                with self.assertRaisesRegex(ValueError, 'seed/design|public initialization'):
                    self.validate()
                self.result['fit'][scope][key] = old

    def test_complete_cell_is_hash_bound_and_cannot_be_overwritten(self):
        self.publish()
        self.assertEqual(self.store.status(0), 'validated')
        before = {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()}
        with self.assertRaises(FileExistsError):
            self.store.start(execution(), 0)
        with self.assertRaises(ValueError):
            self.store.complete(self.result, self.arrays, 0)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()})

    def test_failed_partial_cell_cannot_resume_or_overwrite_evidence(self):
        self.store.initialize()
        self.store.start(execution(), 0)
        self.store.fail(RuntimeError('fixture failure'), 0)
        before = {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()}
        self.assertEqual(set(before), {'started.json', 'failure.json'})
        for operation in [lambda: self.store.status(0), lambda: self.store.start(execution(), 0),
                          lambda: self.store.complete(self.result, self.arrays, 0)]:
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.store.cell_dir(0).iterdir()})

    def test_validated_completion_cannot_be_marked_failed(self):
        self.publish()
        with self.assertRaisesRegex(ValueError, 'completed'):
            self.store.fail(RuntimeError('late failure'), 0)
        self.assertEqual(self.store.status(0), 'validated')
        self.assertFalse((self.store.cell_dir(0) / 'failure.json').exists())

    def test_changed_model_bytes_and_missing_completion_are_not_validated(self):
        self.publish()
        model = self.store.cell_dir(0) / 'model.npz'
        model.write_bytes(b'mutated model')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.store.status(0)
        (self.store.cell_dir(0) / 'done.json').unlink()
        with self.assertRaisesRegex(ValueError, 'partial'):
            self.store.status(0)

    def test_nonempty_run_without_manifest_is_preserved_and_rejected(self):
        self.directory.mkdir()
        witness = self.directory / 'unrelated.txt'
        witness.write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'lacks manifest'):
            self.store.initialize()
        self.assertEqual(witness.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
