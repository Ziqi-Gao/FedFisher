"""Reporting fixtures contain tiny saved arrays, never trained or inferred data."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from effect_modifier_experiment.survival.artifacts import array_hash
from effect_modifier_experiment.survival.links import survival_mean
from effect_modifier_experiment.survival_pooled.evaluation import compare_predictions, summarize_arrays
from reporting.effect_modifier import survival_pooled as reporting


SCRATCH = Path('/scr/del6500/FedFisher/survival-validation')


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


class SurvivalPooledReportingTests(unittest.TestCase):
    def setUp(self):
        # This explicit scratch parent avoids both /tmp and home-directory caches.
        self.temporary = tempfile.TemporaryDirectory(prefix='pooled-report-fixture-', dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_root = self.root / 'runs'
        self.run_root.mkdir()
        self.proof_path = self.root / 'acceptance.json'
        self.units = []
        self.unit_paths = []
        self.truth = dict(eta0=np.array([-.7, .4]), eta1=np.array([-.3, .8]))
        self.truth.update(survival0=survival_mean(self.truth['eta0']),
                          survival1=survival_mean(self.truth['eta1']))
        treatment = np.array([0., 1.])
        predictions = dict(eta0=self.truth['eta0'], eta1=self.truth['eta1'],
                           s0=self.truth['survival0'], s1=self.truth['survival1'])
        metric = summarize_arrays(predictions, self.truth, treatment)
        for family in ('linear', 'shallow', 'deep'):
            historical = ('dslasso',) if family == 'linear' else ('fedfisher_diag', 'fedfisher_kfac')
            tags = historical + ('pooled_site_local_pv', 'pooled_pooled_km_pv')
            for scenario in ('iid', 'covariate_shift'):
                run_id = 'survival_pooled_v1_%s_%s_pilot' % (family, scenario)
                run = self.run_root / run_id
                run.mkdir()
                manifest = run / 'run_manifest.env'
                manifest.write_text('format_version=2\nprotocol=survival-pooled-prediction-v1\n'
                                    + 'run_id=' + run_id + '\n')
                for seed in (0, 49):
                    directory = run / ('seed_%03d' % seed)
                    directory.mkdir()
                    identity = dict(run_id=run_id, cell=family, scenario=scenario, seed=seed,
                                    config_sha256='1' * 64, provenance_sha256='2' * 64)
                    started = directory / 'started.json'
                    result_path = directory / 'result.json'
                    model_path = directory / 'model.npz'
                    done_path = directory / 'done.json'
                    write_json(started, dict(identity=identity, execution=dict(attempt=1)))
                    arrays = {'truth__' + key: value.copy() for key, value in self.truth.items()}
                    arrays['factual_a'] = treatment.copy()
                    for tag in tags:
                        arrays.update({tag + '__prediction__' + key: value.copy()
                                       for key, value in predictions.items()})
                    np.savez(model_path, **arrays)
                    result = dict(identity=identity, status='complete',
                                  prediction={tag: copy.deepcopy(metric) for tag in tags},
                                  comparisons={tag: {scope: compare_predictions(predictions, predictions)
                                      for scope in ('site_local_pv', 'pooled_km_pv')} for tag in historical},
                                  prediction_arrays_sha256=array_hash(arrays))
                    write_json(result_path, result)
                    write_json(done_path, dict(identity=identity, artifacts={
                        path.name: reporting._binding(path)['sha256']
                        for path in (started, model_path, result_path)}))
                    files = [started, model_path, result_path, done_path]
                    self.units.append(dict(family=family, scenario=scenario, seed=seed,
                        run_id=run_id, manifest=reporting._binding(manifest),
                        artifacts=[reporting._binding(path) for path in files]))
                    self.unit_paths.append(dict(started=started, model=model_path,
                                                result=result_path, done=done_path))
        self.proof = dict(schema_version=1, protocol='survival-pooled-prediction-v1',
                          scope='survival_pooled_pilot_scientific_acceptance',
                          passed=True, inference_performed=False,
                          validated_unit_count=12, units=self.units)
        write_json(self.proof_path, self.proof)
        self.run_patch = patch.object(reporting, 'RUN_ROOT', self.run_root)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)

    def refresh(self, index=0):
        """Rebind intentionally altered fixture evidence to test semantic checks."""
        paths = self.unit_paths[index]
        done = json.loads(paths['done'].read_text())
        done['artifacts'] = {paths[key].name: reporting._binding(paths[key])['sha256']
                             for key in ('started', 'model', 'result')}
        write_json(paths['done'], done)
        self.units[index]['artifacts'] = [reporting._binding(paths[key])
                                         for key in ('started', 'model', 'result', 'done')]
        write_json(self.proof_path, self.proof)

    def test_completed_saved_arrays_load_without_generator_fit_or_inference(self):
        with patch('effect_modifier_experiment.survival.data.generate_survival',
                   side_effect=AssertionError('reporting must not generate data')), \
             patch('effect_modifier_experiment.survival_pooled.methods.fit_pooled',
                   side_effect=AssertionError('reporting must not fit models')):
            proof, rows = reporting.load_completed(self.proof_path)
        self.assertEqual(proof, self.proof)
        self.assertEqual(len(rows), 44)  # four linear units*3 + eight neural units*4
        self.assertTrue(all(row['summary']['combined']['rmse'] == 0 for row in rows))
        summary = reporting.aggregate(rows)
        self.assertEqual(len(summary), 22)
        self.assertTrue(all(row['seeds'] == 2 for row in summary))
        gaps = reporting.aggregate_gaps(rows)
        self.assertEqual(len(gaps), 20)  # five FL methods, two scenarios, two PV scopes
        self.assertTrue(all(row['combined_rms_gap_mean'] == 0 for row in gaps))

    def test_paired_rms_gap_mean_standard_error_and_arm_mapping(self):
        rows = []
        reference = dict(eta0=np.log(-np.log(np.array([.2, .4]))),
                         eta1=np.log(-np.log(np.array([.3, .5]))))
        reference.update(s0=survival_mean(reference['eta0']), s1=survival_mean(reference['eta1']))
        for seed, offset in ((0, .1), (49, .2)):
            changed = dict(eta0=np.log(-np.log(reference['s0'] + offset)),
                           eta1=np.log(-np.log(reference['s1'] + 2 * offset)))
            changed.update(s0=survival_mean(changed['eta0']), s1=survival_mean(changed['eta1']))
            gap = compare_predictions(changed, reference)
            rows.append(dict(family='linear', scenario='iid', seed=seed,
                             method='dslasso', gaps={'site_local_pv': gap}))
        result = reporting.aggregate_gaps(rows)[0]
        self.assertAlmostEqual(result['control_rms_gap_mean'], .15)
        self.assertAlmostEqual(result['control_rms_gap_se'], .05)
        self.assertAlmostEqual(result['treatment_rms_gap_mean'], .3)
        self.assertAlmostEqual(result['treatment_rms_gap_se'], .1)
        self.assertAlmostEqual(result['combined_rms_gap_mean'], .15 * np.sqrt(2.5))
        self.assertAlmostEqual(result['combined_rms_gap_se'], .05 * np.sqrt(2.5))

    def test_aggregate_means_standard_errors_and_arm_mapping(self):
        rows = []
        for seed, offset in ((0, 0.), (49, .2)):
            summary = {}
            for target, baseline in (('combined', .2), ('factual', .3), ('contrast', 1.),
                                     ('control', .1), ('treatment', .4)):
                value = dict(rmse=baseline + offset, mae=baseline / 2 + offset,
                             bias=-.1 + 2 * offset)
                if target in ('control', 'treatment'):
                    summary.setdefault('arms', {})['0' if target == 'control' else '1'] = value
                else:
                    summary[target] = value
            rows.append(dict(family='linear', scenario='iid', seed=seed,
                             method='dslasso', summary=summary))
        result = reporting.aggregate(rows)[0]
        self.assertAlmostEqual(result['combined_rmse_mean'], .3)
        self.assertAlmostEqual(result['combined_rmse_se'], .1)
        self.assertAlmostEqual(result['combined_bias_mean'], .1)
        self.assertAlmostEqual(result['combined_bias_se'], .2)
        self.assertAlmostEqual(result['control_rmse_mean'], .2)
        self.assertAlmostEqual(result['treatment_rmse_mean'], .5)
        self.assertAlmostEqual(result['contrast_rmse_mean'], 1.1)
        with self.assertRaises(ValueError):
            reporting.aggregate(rows + [copy.deepcopy(rows[0])])

    def test_unpassed_inference_or_wrong_count_proofs_are_rejected(self):
        for key, value in (('passed', False), ('inference_performed', True),
                           ('validated_unit_count', 13), ('protocol', 'another-protocol')):
            with self.subTest(key=key):
                altered = copy.deepcopy(self.proof)
                altered[key] = value
                write_json(self.proof_path, altered)
                with self.assertRaises(ValueError):
                    reporting.load_completed(self.proof_path)

    def test_changed_artifact_hash_is_rejected(self):
        self.unit_paths[0]['result'].write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            reporting.load_completed(self.proof_path)

    def test_done_identity_mismatch_is_rejected_with_valid_file_hashes(self):
        path = self.unit_paths[0]['done']
        done = json.loads(path.read_text())
        done['identity']['seed'] = 49
        write_json(path, done)
        self.refresh()
        with self.assertRaises(ValueError):
            reporting.load_completed(self.proof_path)

    def test_wrong_started_identity_is_rejected_even_when_done_hashes_match(self):
        path = self.unit_paths[0]['started']
        started = json.loads(path.read_text())
        started['identity']['cell'] = 'deep'
        write_json(path, started)
        self.refresh()
        with self.assertRaises(ValueError):
            reporting.load_completed(self.proof_path)

    def test_rebound_wrong_metric_and_saved_array_digest_are_rejected(self):
        path = self.unit_paths[0]['result']
        original = json.loads(path.read_text())
        for field in ('metric', 'digest'):
            with self.subTest(field=field):
                result = copy.deepcopy(original)
                if field == 'metric':
                    result['prediction']['dslasso']['factual']['rmse'] = .1
                else:
                    result['prediction_arrays_sha256'] = '0' * 64
                write_json(path, result)
                self.refresh()
                with self.assertRaises(ValueError):
                    reporting.load_completed(self.proof_path)

    def test_duplicate_artifact_inventory_is_rejected(self):
        self.units[0]['artifacts'].append(copy.deepcopy(self.units[0]['artifacts'][0]))
        write_json(self.proof_path, self.proof)
        with self.assertRaises(ValueError):
            reporting.load_completed(self.proof_path)

    def test_empty_or_duplicate_scientific_units_are_rejected(self):
        for units in ([], self.units + [copy.deepcopy(self.units[0])]):
            with self.subTest(unit_count=len(units)):
                altered = copy.deepcopy(self.proof)
                altered['units'] = units
                altered['validated_unit_count'] = len(units)
                write_json(self.proof_path, altered)
                with self.assertRaises(ValueError):
                    reporting.load_completed(self.proof_path)

    def test_incomplete_model_family_coverage_is_rejected(self):
        path = self.unit_paths[0]['result']
        result = json.loads(path.read_text())
        result['prediction'].pop('pooled_site_local_pv')
        write_json(path, result)
        self.refresh()
        with self.assertRaises(ValueError):
            reporting.load_completed(self.proof_path)

    def test_rebound_comparison_metrics_are_rejected(self):
        path = self.unit_paths[0]['result']
        result = json.loads(path.read_text())
        result['comparisons']['dslasso']['site_local_pv']['combined']['rmse'] = .25
        write_json(path, result)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'gaps differ'):
            reporting.load_completed(self.proof_path)

    def test_scientific_run_identity_must_match_acceptance_unit(self):
        paths = self.unit_paths[0]
        for key in ('started', 'result', 'done'):
            record = json.loads(paths[key].read_text())
            record['identity']['run_id'] = 'survival_pooled_v1_deep_iid_pilot'
            write_json(paths[key], record)
        self.refresh()
        with self.assertRaises(ValueError):
            reporting.load_completed(self.proof_path)

    def test_artifacts_outside_run_root_are_rejected(self):
        outside = self.root / 'foreign_manifest.env'
        outside.write_text('format_version=2\n')
        self.units[0]['manifest'] = reporting._binding(outside)
        write_json(self.proof_path, self.proof)
        with self.assertRaises(ValueError):
            reporting.load_completed(self.proof_path)

    def test_acceptance_proof_change_during_read_is_rejected(self):
        original_binding = reporting._binding
        seen = 0

        def changed_at_end(path):
            nonlocal seen
            binding = original_binding(path)
            if Path(path) == self.proof_path:
                seen += 1
                if seen > 1:
                    binding['sha256'] = '0' * 64
            return binding

        with patch.object(reporting, '_binding', side_effect=changed_at_end):
            with self.assertRaisesRegex(ValueError, 'proof changed'):
                reporting.load_completed(self.proof_path)


if __name__ == '__main__':
    unittest.main()
