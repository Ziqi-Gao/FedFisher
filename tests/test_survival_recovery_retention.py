"""Returned fits remain inspectable after validation/evaluation rejection."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.survival import artifacts, evaluation, recovery, recovery_method
import test_survival_recovery_contract as fixture


class PostFitRetentionTests(unittest.TestCase):
    def setUp(self):
        root = Path('/scr/del6500/FedFisher/survival-repair-validation/retention')
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(artifacts, 'current_provenance',
                                                   return_value=fixture.provenance()))

    def store(self, config):
        prerequisite = None
        if not recovery.is_diagnostic(config['task']):
            prerequisite = dict(diagnostic_acceptance_sha256='a' * 64,
                provenance_sha256=fixture.provenance()['provenance_sha256'],
                selected_recipe=config['recovery_recipe'])
        with mock.patch.object(recovery, 'run_config', return_value=deepcopy(config)):
            store = recovery.RecoveryStore(config, fixture.provenance(),
                Path(self.temp.name) / config['run_id'], prerequisite=prerequisite)
        store.initialize()
        store.start(fixture.execution(), 0)
        return store

    def assert_saved_failure(self, store, error):
        store.fail(error, 0)
        path = store.cell_dir(0)
        saved = json.loads((path / 'failure.json').read_text())
        self.assertEqual(saved['message'], str(error))
        self.assertEqual(saved['failure_stage'], error.stage)
        self.assertEqual(saved['original_error_type'], error.original_error_type)
        self.assertFalse(saved['scientific_completion'])
        self.assertEqual(saved['optimization_diagnostics'], error.diagnostics)
        self.assertIn('terminal_gradient', saved['optimization_diagnostics']['paths'][0]['unpenalized_start'])
        payload = (path / 'failure_arrays.npz').read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), saved['failure_arrays']['sha256'])
        with np.load(path / 'failure_arrays.npz', allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        self.assertEqual(artifacts.array_hash(arrays), saved['failure_arrays']['arrays_sha256'])
        for name in error.arrays:
            np.testing.assert_array_equal(arrays[name], error.arrays[name])
        self.assertFalse((path / store.done_name).exists())
        with self.assertRaisesRegex(ValueError, 'partial'):
            store.status(0)
        before = {p.name: p.read_bytes() for p in path.iterdir()}
        with self.assertRaises(FileExistsError):
            store.fail(error, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
        return saved

    def test_post_fit_validation_retains_paths_model_and_failed_stage_timing(self):
        config = fixture.tiny_config()
        with mock.patch.object(recovery_method, 'validate_recovery_diagnostics',
                               side_effect=ValueError('recovery accepted a majorization violation')):
            with self.assertRaises(recovery.FitValidationFailure) as caught:
                recovery.compute_cell(config, 0)
        error = caught.exception
        self.assertEqual(error.stage, 'post_fit_validation')
        self.assertEqual(set(error.arrays), {'intercept', 'coefficients'})
        saved = self.assert_saved_failure(self.store(config), error)
        self.assertGreater(saved['saved_evidence']['timing_stages']['fit']['wall_seconds'], 0)

    def test_evaluation_exception_retains_the_completed_fit(self):
        config = fixture.tiny_config(formal=True)
        with mock.patch.object(evaluation, 'evaluate_model', side_effect=ValueError('evaluation rejected')):
            with self.assertRaises(recovery.FitValidationFailure) as caught:
                recovery.compute_cell(config, 0)
        error = caught.exception
        self.assertEqual(error.stage, 'evaluation')
        saved = self.assert_saved_failure(self.store(config), error)
        self.assertGreater(saved['saved_evidence']['timing_stages']['evaluation']['wall_seconds'], 0)

    def test_store_validation_rejection_is_preserved_by_the_runner(self):
        from effect_modifier_experiment.survival import recovery_acceptance as gate
        config = fixture.tiny_config()
        result, arrays = recovery.compute_cell(config, 0)
        with mock.patch.object(recovery, 'run_config', return_value=deepcopy(config)):
            store = recovery.RecoveryStore(config, fixture.provenance(),
                                           Path(self.temp.name) / config['run_id'])
        job = dict(fixture.execution(), task=recovery.DIAGNOSTIC_TASK,
                   parameters=dict(case='iid_000', recipe='baseline'))
        with mock.patch.object(gate, 'check_original_closeout'), \
                mock.patch.object(gate, 'original_reference'), \
                mock.patch.object(recovery, 'run_config', return_value=config), \
                mock.patch.object(recovery, 'RecoveryStore', return_value=store), \
                mock.patch.object(recovery, 'compute_cell', return_value=(result, arrays)), \
                mock.patch.object(store, '_validate_result', side_effect=ValueError('saved result rejected')), \
                mock.patch.dict(os.environ, {'SERVER_SCHEDULER_ATTEMPT': '1'}):
            with self.assertRaises(recovery.FitValidationFailure):
                recovery.run_job(job)
        saved = json.loads((store.cell_dir(0) / 'failure.json').read_text())
        self.assertEqual(saved['failure_stage'], 'artifact_publication')
        self.assertEqual(saved['optimization_diagnostics'], result['fit'])
        self.assertTrue((store.cell_dir(0) / 'failure_arrays.npz').exists())
        self.assertFalse((store.cell_dir(0) / store.done_name).exists())

    def test_required_v2_evidence_cannot_be_downgraded_to_legacy_schema(self):
        config = fixture.tiny_config()
        config['required_evidence_schema'] = recovery_method.RECOVERY_SCHEMA_V2
        result, arrays = recovery.compute_cell(config, 0)
        result['identity'] = dict(seed=0)
        result['fit']['recovery_schema'] = recovery_method.RECOVERY_SCHEMA_V1
        with self.assertRaises(ValueError):
            recovery.validate_result(result, arrays, result['identity'], config)

    def repair_config(self, formal=False, epoch=2):
        tasks = recovery.tasks_for_epoch(epoch)
        config = recovery.run_config(
            tasks[1 if formal else 0],
            dict(scenario='iid', shard_index=0, shard_count=16) if formal else dict(case='iid_000'),
            recipe=recovery.REPAIR_RECIPE)
        small = fixture.tiny_config(formal=formal)
        config['dgp'] = small['dgp']
        config['evaluation'] = small['evaluation']
        config['method']['optimizer'] = small['method']['optimizer']
        return config

    def test_real_tiny_repair_pilot_and_formal_outputs_validate_and_skip(self):
        for epoch in (2, 3):
            with self.subTest(epoch=epoch):
                self.check_real_tiny_epoch(epoch)

    def check_real_tiny_epoch(self, epoch):
        for formal in (False, True):
            with self.subTest(formal=formal):
                config = self.repair_config(formal, epoch=epoch)
                result, arrays = recovery.compute_cell(config, 0)
                store = self.store(config)
                result['identity'] = store.identity(0)
                store.complete(result, arrays, 0)
                self.assertEqual(store.status(0), 'validated')
                self.assertEqual(result['fit']['recovery_schema'], recovery_method.RECOVERY_SCHEMA_V2)
                self.assertEqual('evaluation' in result, formal)
                before = {p.name: p.read_bytes() for p in store.cell_dir(0).iterdir()}
                with self.assertRaises(FileExistsError):
                    store.start(dict(fixture.execution(), attempt=2), 0)
                self.assertEqual(before, {p.name: p.read_bytes() for p in store.cell_dir(0).iterdir()})

    def test_repair_pilot_iteration_limit_is_failed_not_a_completed_observation(self):
        for epoch in (2, 3):
            with self.subTest(epoch=epoch):
                self.check_iteration_limit(epoch)

    def check_iteration_limit(self, epoch):
        config = self.repair_config(epoch=epoch)
        config['method']['optimizer'].update(max_iter=1, tolerance=1e-12)
        with self.assertRaises(recovery_method.OptimizationFailure) as caught:
            recovery.compute_cell(config, 0)
        store = self.store(config)
        store.fail(caught.exception, 0)
        saved = json.loads((store.cell_dir(0) / 'failure.json').read_text())
        self.assertEqual(saved['optimization_diagnostics']['outcome'], 'iteration_limit')
        self.assertFalse((store.cell_dir(0) / store.done_name).exists())
        with self.assertRaisesRegex(ValueError, 'partial'):
            store.status(0)


if __name__ == '__main__':
    unittest.main()
