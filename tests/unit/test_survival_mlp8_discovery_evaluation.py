"""Analytical, bounded discovery fixtures; no survival simulation or training."""

import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from effect_modifier_experiment.survival import evaluation as original
from effect_modifier_experiment.survival_mlp8.discovery_evaluation import (
    evaluate_discovery, validate_discovery,
)
from effect_modifier_experiment.survival_mlp8.protocol import TAGS
from effect_modifier_experiment.survival_pooled.evaluation import probability_predictions


def dataset():
    x = np.array([[-2., 1., 4., 0.], [-1., -2., 0., 3.],
                  [0., 2., -3., -1.], [1., -1., 2., 2.],
                  [2., 0., 1., -2.], [3., 3., -1., 1.]])
    return SimpleNamespace(evaluation=SimpleNamespace(x=x),
        evaluation_truth=SimpleNamespace(delta=2. * (x[:, 0] - x[:, 1]),
                                         modifier_mask=np.array([True, True, False, False])))


def predictor(scale=1., prognostic=True, designs=None):
    def predict(design):
        a, x, ax = design[:, 0], design[:, 1:5], design[:, 5:9]
        np.testing.assert_array_equal(ax, a[:, None] * x)
        if designs is not None:
            designs.append(design.copy())
        main = 7. * x[:, 2] + 3. * x[:, 3] if prognostic else np.zeros(len(x))
        return main + scale * 2. * (ax[:, 0] - ax[:, 1])
    return predict


def evidence(seed=49, data=None):
    data = dataset() if data is None else data
    config = dict(dgp=dict(covariate_dim=4, num_test=6), seeds=[0, 49],
        evaluation=dict(modifier_permutation_performed=True,
            discovery=dict(primary=original.PRIMARY, repeats=5,
                top2='probability_all_signals_selected', auroc=True,
                seed_rule='SeedSequence([seed,4291,repeat,covariate])', tie_rule=original.TIE_RULE)))
    reference = dict(identity=dict(seed=seed),
        evaluation=original.evaluate_model(predictor(), data, seed=seed))
    arrays = {'discovery__x': data.evaluation.x.copy(),
              'truth__delta': data.evaluation_truth.delta.copy(),
              'truth__modifier_mask': data.evaluation_truth.modifier_mask.copy()}
    summaries = {}
    for tag, scale in zip(TAGS, (1., .5, 0., 1.5)):
        predict = predictor(scale)
        summaries[tag] = evaluate_discovery(predict, data, seed)
        predictions = probability_predictions(predict, data)
        arrays.update({tag + '__prediction__' + key: value
                       for key, value in predictions.items()})
    return summaries, arrays, config, reference


class MLP8DiscoveryTests(unittest.TestCase):
    def test_delegates_the_unchanged_original_evaluator(self):
        data, predict = dataset(), predictor()
        expected = original.evaluate_model(predict, data, repeats=5, seed=49)
        with patch.object(original, 'evaluate_model', wraps=original.evaluate_model) as delegated:
            actual = evaluate_discovery(predict, data, 49)
        delegated.assert_called_once_with(predict, data, repeats=5, seed=49)
        self.assertEqual(actual, expected)
        self.assertEqual(actual['repeats'], 5)

    def test_true_modifiers_recover_and_prognostic_terms_cancel(self):
        data = dataset()
        with_main = evaluate_discovery(predictor(), data, 49)
        without_main = evaluate_discovery(predictor(prognostic=False), data, 49)
        self.assertEqual(with_main, without_main)
        scores = np.asarray(with_main['importance_by_repeat'])
        self.assertTrue((scores[:, :2] > 0).all())
        np.testing.assert_array_equal(scores[:, 2:], np.zeros((5, 2)))
        self.assertEqual(with_main['contrast_rmse'], 0.)
        self.assertEqual(with_main['recovery']['probability_all_signals_selected'], 1.)
        self.assertEqual(with_main['recovery']['auroc'], 1.)

    def test_every_intervention_rebuilds_raw_x_and_treatment_interactions(self):
        data, designs = dataset(), []
        evaluate_discovery(predictor(designs=designs), data, 49)
        self.assertEqual(len(designs), 2 * (1 + 5 * 4))
        for index in range(0, len(designs), 2):
            control, treated = designs[index:index + 2]
            np.testing.assert_array_equal(control[:, 0], np.zeros(6))
            np.testing.assert_array_equal(treated[:, 0], np.ones(6))
            np.testing.assert_array_equal(control[:, 1:5], treated[:, 1:5])
            np.testing.assert_array_equal(control[:, 5:], np.zeros((6, 4)))
            np.testing.assert_array_equal(treated[:, 5:], treated[:, 1:5])
        # Each original stream perturbs just its raw feature.  The untreated
        # and treated designs must both contain that same perturbed feature.
        for repeat in range(5):
            for j in range(4):
                rng = np.random.default_rng(np.random.SeedSequence([49, 4291, repeat, j]))
                changed = data.evaluation.x.copy()
                changed[:, j] = changed[rng.permutation(6), j]
                np.testing.assert_array_equal(designs[2 + 2 * (repeat * 4 + j)][:, 1:5], changed)

    def test_common_seed_streams_pair_all_models_and_ignore_prognosis(self):
        summaries, arrays, config, reference = evidence()
        self.assertEqual(len({summary['permutation_sha256'] for summary in summaries.values()}), 1)
        self.assertEqual(next(iter(summaries.values()))['permutation_sha256'],
                         reference['evaluation']['permutation_sha256'])
        other_seed = evaluate_discovery(predictor(), dataset(), 0)
        self.assertNotEqual(other_seed['permutation_sha256'], reference['evaluation']['permutation_sha256'])
        self.assertTrue(validate_discovery(summaries, arrays, config, reference, seed=49))

    def test_zero_signal_and_exact_boundary_ties_use_original_probabilities(self):
        zero = evaluate_discovery(predictor(scale=0.), dataset(), 49)
        self.assertEqual(zero['importance'], [0.] * 4)
        self.assertEqual(zero['recovery']['probability_all_signals_selected'], 1. / math.comb(4, 2))
        self.assertEqual(zero['recovery']['auroc'], .5)
        tied = original.signal_recovery([3., 2., 2., 2.], np.array([True, True, False, False]))
        self.assertEqual(tied['probability_all_signals_selected'], 1. / 3.)
        self.assertEqual(tied['boundary_tie_size'], 3)
        self.assertEqual(tied['boundary_slots'], 1)
        # Top-2 is joint recovery probability, rather than average precision.
        self.assertNotEqual(tied['expected_topk_precision'], tied['probability_all_signals_selected'])

    def test_saved_validation_invokes_no_inference_generator_or_rng(self):
        summaries, arrays, config, reference = evidence()
        before = copy.deepcopy(summaries), {k: v.copy() for k, v in arrays.items()}
        with patch.object(original, 'evaluate_model', side_effect=AssertionError('inference')), \
             patch.object(original, 'treatment_contrast', side_effect=AssertionError('inference')), \
             patch('numpy.random.default_rng', side_effect=AssertionError('RNG')), \
             patch('effect_modifier_experiment.survival.data.generate_survival', side_effect=AssertionError('generator')):
            self.assertTrue(validate_discovery(summaries, arrays, config, reference, seed=49))
        self.assertEqual(summaries, before[0])
        for name in arrays:
            np.testing.assert_array_equal(arrays[name], before[1][name])

    def test_saved_baseline_hash_and_direct_oracle_delta_bind_error(self):
        data = dataset()
        data.evaluation_truth.delta = np.linspace(.1, .6, 6)
        summaries, arrays, config, reference = evidence(data=data)
        # The generator preserves delta explicitly.  Subtracting very large
        # oracle predictors can lose it, so this arithmetic must use delta.
        arrays['truth__eta0'] = np.full(6, float(2 ** 54))
        arrays['truth__eta1'] = arrays['truth__eta0'] + arrays['truth__delta']
        rounded_truth = arrays['truth__eta1'] - arrays['truth__eta0']
        self.assertFalse(np.array_equal(rounded_truth, arrays['truth__delta']))
        self.assertTrue(validate_discovery(summaries, arrays, config, reference, seed=49))
        changed = copy.deepcopy(summaries)
        baseline = arrays[TAGS[0] + '__prediction__eta1'] - arrays[TAGS[0] + '__prediction__eta0']
        wrong_error = baseline - rounded_truth
        changed[TAGS[0]]['contrast_squared_error_sum'] = float(wrong_error @ wrong_error)
        changed[TAGS[0]]['contrast_rmse'] = float(np.sqrt((wrong_error @ wrong_error) / 6))
        original.validate_evaluation(changed[TAGS[0]], 4, 6, 5)
        with self.assertRaisesRegex(ValueError, 'saved eta predictions'):
            validate_discovery(changed, arrays, config, reference, seed=49)

    def test_rehashed_baseline_prediction_tamper_cannot_change_discovery_endpoint(self):
        summaries, arrays, config, reference = evidence()
        changed = {k: v.copy() for k, v in arrays.items()}
        changed[TAGS[0] + '__prediction__eta1'][0] += .25
        with self.assertRaisesRegex(ValueError, 'saved eta predictions'):
            validate_discovery(summaries, changed, config, reference, seed=49)
        summaries = copy.deepcopy(summaries)
        new_baseline = changed[TAGS[0] + '__prediction__eta1'] - changed[TAGS[0] + '__prediction__eta0']
        summaries[TAGS[0]]['estimated_contrast_sha256'] = original._array_hash(new_baseline)
        with self.assertRaisesRegex(ValueError, 'saved eta predictions'):
            validate_discovery(summaries, changed, config, reference, seed=49)

    def test_covariate_truth_mask_permutation_and_seed_tampering_fail_closed(self):
        summaries, arrays, config, reference = evidence()
        for name in ('discovery__x', 'truth__delta', 'truth__modifier_mask'):
            changed = {key: value.copy() for key, value in arrays.items()}
            changed[name].flat[0] = not changed[name].flat[0] if name.endswith('mask') else changed[name].flat[0] + 1.
            with self.subTest(array=name), self.assertRaises(ValueError):
                validate_discovery(summaries, changed, config, reference, seed=49)
        for field in ('raw_x_sha256', 'contrast_truth_sha256', 'permutation_sha256', 'estimated_contrast_sha256'):
            changed = copy.deepcopy(summaries)
            changed[TAGS[0]][field] = '0' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_discovery(changed, arrays, config, reference, seed=49)
        for bad_seed in (0, True, -1, 48):
            with self.subTest(seed=bad_seed), self.assertRaises(ValueError):
                validate_discovery(summaries, arrays, config, reference, seed=bad_seed)

    def test_scores_and_recovery_are_recomputed_from_json(self):
        summaries, arrays, config, reference = evidence()
        for field in ('importance', 'importance_by_repeat'):
            changed = copy.deepcopy(summaries)
            target = changed[TAGS[0]][field]
            (target[0] if field == 'importance_by_repeat' else target)[0] += .01
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_discovery(changed, arrays, config, reference)
        changed = copy.deepcopy(summaries)
        changed[TAGS[0]]['recovery']['probability_all_signals_selected'] = .5
        with self.assertRaises(ValueError):
            validate_discovery(changed, arrays, config, reference)

    def test_inventory_types_dimensions_and_fixed_recipe_are_strict(self):
        summaries, arrays, config, reference = evidence()
        for mutation in ('extra_tag', 'missing_tag', 'extra_summary', 'extra_recovery',
                         'bool_count', 'bool_slots', 'bool_metric', 'string_score', 'bool_score'):
            changed = copy.deepcopy(summaries)
            if mutation == 'extra_tag':
                changed['historical_dslasso'] = changed[TAGS[0]]
            elif mutation == 'missing_tag':
                del changed[TAGS[0]]
            elif mutation == 'extra_summary':
                changed[TAGS[0]]['extra'] = 1
            elif mutation == 'extra_recovery':
                changed[TAGS[0]]['recovery']['extra'] = 1
            elif mutation == 'bool_count':
                changed[TAGS[0]]['repeats'] = True
            elif mutation == 'bool_slots':
                changed[TAGS[0]]['recovery']['boundary_slots'] = True
            elif mutation == 'string_score':
                changed[TAGS[0]]['importance_by_repeat'][0][0] = '1.0'
            elif mutation == 'bool_score':
                changed[TAGS[0]]['importance_by_repeat'][0][0] = True
            else:
                changed[TAGS[0]]['recovery']['auroc'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_discovery(changed, arrays, config, reference)
        for name, value in (('discovery__x', arrays['discovery__x'].astype(np.float32)),
                            ('discovery__x', arrays['discovery__x'][:2]),
                            ('truth__delta', np.full(6, np.nan)),
                            ('truth__modifier_mask', arrays['truth__modifier_mask'].astype(int))):
            changed = dict(arrays, **{name: value})
            with self.subTest(array=name), self.assertRaises(ValueError):
                validate_discovery(summaries, changed, config, reference)
        for key, value in (('repeats', 4), ('auroc', 1), ('seed_rule', 'another_stream'),
                            ('top2', 'expected_topk_precision')):
            changed = copy.deepcopy(config)
            changed['evaluation']['discovery'][key] = value
            with self.subTest(recipe=key), self.assertRaises(ValueError):
                validate_discovery(summaries, arrays, changed, reference)


if __name__ == '__main__':
    unittest.main()
