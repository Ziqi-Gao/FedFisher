"""Pure completed-record fixtures: no dataset generation, fitting, or inference."""
from copy import deepcopy
import hashlib
import unittest
from unittest import mock

from effect_modifier_experiment.one_shot_effects import protocol, reporting
from effect_modifier_experiment.one_shot_effects.wire import canonical


def bind_identity(record):
    config = record['config']
    record['result']['identity'] = dict(
        **{key: config[key] for key in ('recipe', 'dataset', 'run_id', 'seed')},
        config_sha256=hashlib.sha256(canonical(config).encode()).hexdigest(),
        provenance_sha256='a' * 64)


def fixture_records(dataset='scmv2_base'):
    records = []
    for index, compression in enumerate(reporting.COMPRESSIONS):
        config = protocol.run_config(protocol.PILOT_TASK, dict(
            recipe='compression_shallow_' + compression + '_r16', dataset=dataset))
        budget = protocol.compression_wire_budget(config)
        weights = [0.4, 0.6]
        applicable = dataset != 'scmv2_null'
        recovery = dict(applicable=applicable, auroc=0.75 if applicable else None,
            expected_topk_hits=1.0 if applicable else None)
        primary = dict(kind='outcome_mlp', model_sha256='b' * 64,
            metadata=dict(model_name='SyntheticMLP', covariate_dim=100),
            fit=dict(site_weights=weights),
            test=dict(recovery=recovery, cate_rmse=0.125, scores=[0.1, 0.2]),
            holdout=dict(target_weights=weights, rows=[2, 3], permutation_sha256=['c'*64, 'd'*64],
                exact_squared_scores=[0.01, 0.02], summary_squared_scores=[0.011, 0.019],
                recovery=recovery, summary_comparison=dict(maximum_absolute_error=.001*(index+1),
                    mean_absolute_error=.001, spearman_rank_correlation=1.,
                    top2_set_agreement_probability=1.)))
        result = dict(status='complete', models=dict(primary=primary),
            data_fingerprints=dict(sites=[dict(site_id=i, n_total=n,
                counts=dict(train=n-h-1, validation=1, holdout=h),
                train='e'*64, validation='f'*64, holdout='0'*64, partition='1'*64)
                for i, (n, h) in enumerate(((10, 2), (15, 3)))],
                target_weights=weights, test='2'*64, test_rows=20),
            communication=dict(upload_rounds=1, upload_messages=2, upload_bytes=budget*2,
                native_upload_bytes=budget*2-1000-index, padding_bytes=1000+index,
                per_site_upload_bytes=[budget]*2, wire_budget_per_site=budget,
                network_time_measured=False, protocol_distribution_bytes=400,
                final_model_distribution_bytes=600),
            timing=dict(stages={stage:dict(wall_seconds=float(i+1), cpu_seconds=float(i+1)/2,
                failed_calls=0) for i, stage in enumerate(reporting.STAGES)}),
            diagnostics=dict(linearization=dict(score_agreement=dict(low_rank={}),
                tau_rmse=dict(full_gradient=.05, low_rank=.06+index*.01),
                recovery=dict(low_rank=recovery))))
        record = dict(config=config, result=result)
        bind_identity(record)
        records.append(record)
    return records


class OneShotEffectsReportingTests(unittest.TestCase):
    def test_paired_summary_is_ordered_detached_and_never_computes_models(self):
        records = fixture_records()
        before = canonical(records)
        with mock.patch('effect_modifier_experiment.one_shot_effects.fedfisher_em.fit_local',
                        side_effect=AssertionError('report cannot fit')), \
             mock.patch('effect_modifier_experiment.one_shot_effects.fisher_models.predict_tau',
                        side_effect=AssertionError('report cannot infer')):
            result = reporting.compare_compression_records(records[::-1])
        self.assertEqual([row['method'] for row in result['rows']], list(reporting.COMPRESSIONS))
        self.assertEqual(result['target_weights'], [.4, .6])
        self.assertEqual(result['rows'][0]['timing']['method_fit']['wall_seconds'], 9.)
        self.assertEqual(result['rows'][0]['timing']['method_including_evaluation']['wall_seconds'], 14.)
        self.assertEqual(len({row['wire_upload_bytes'] for row in result['rows']}), 1)
        self.assertEqual(len({row['native_upload_bytes'] for row in result['rows']}), 3)
        self.assertEqual(result['rows'][0]['score_errors']['maximum_absolute_error'], .001)
        result['rows'][0]['recovery']['external_test']['auroc'] = 0
        self.assertEqual(canonical(records), before)

    def test_null_recovery_remains_inapplicable(self):
        result = reporting.compare_compression_records(fixture_records('scmv2_null'))
        for row in result['rows']:
            self.assertIsNone(row['recovery']['summary_holdout']['auroc'])
            self.assertIs(row['recovery']['summary_holdout']['applicable'], False)

    def test_missing_duplicate_partial_and_changed_comparisons_fail(self):
        records = fixture_records()
        for bad in (records[:2], [records[0], records[0], records[2]]):
            with self.assertRaises(ValueError):
                reporting.compare_compression_records(bad)
        mutations = (
            lambda r: r['result'].__setitem__('status', 'failed'),
            lambda r: r['config']['fisher'].__setitem__('learning_rate', .2),
            lambda r: r['config']['evaluation'].__setitem__('repeats', 4),
            lambda r: r['result']['data_fingerprints'].__setitem__('test', '3'*64),
            lambda r: r['result']['data_fingerprints']['sites'][0].__setitem__('train', '3'*64),
            lambda r: r['result']['models']['primary'].__setitem__('model_sha256', '4'*64),
            lambda r: r['result']['models']['primary']['holdout']['permutation_sha256'].__setitem__(0, '5'*64),
            lambda r: r['result']['models']['primary']['fit'].__setitem__('site_weights', [.5, .5]),
            lambda r: r['result']['communication'].__setitem__('wire_budget_per_site', 1),
            lambda r: r['result']['communication'].__setitem__('native_upload_bytes', 1),
            lambda r: r['result']['timing']['stages']['train'].__setitem__('wall_seconds', float('nan')),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                bad = deepcopy(records)
                mutation(bad[1])
                bind_identity(bad[1])
                with self.assertRaises(ValueError):
                    reporting.compare_compression_records(bad)
        bad = deepcopy(records)
        bad[1]['result']['identity']['provenance_sha256'] = '9'*64
        with self.assertRaises(ValueError):
            reporting.compare_compression_records(bad)


if __name__ == '__main__':
    unittest.main()
