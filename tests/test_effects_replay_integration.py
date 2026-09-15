"""Tiny composition fixture through replay computation and its full validator."""
import copy
import unittest
from unittest import mock

import torch

from effect_modifier_experiment.one_shot_effects import fedfisher_em, model_io
from effect_modifier_experiment.one_shot_effects import protocol as original_protocol
from effect_modifier_experiment.one_shot_effects import runner as original_runner
from effect_modifier_experiment.one_shot_effects.diagnostics import protocol, runner, validation


class ReplayIntegrationTests(unittest.TestCase):
    def test_tiny_checkpoint_replay_and_full_readonly_semantic_validation(self):
        torch.set_num_threads(1)
        sources = {}
        for role in ('aggregation','linearity'):
            config = original_protocol.run_config(original_protocol.PILOT_TASK,
                dict(recipe=role+'_shallow',dataset='scmv2_base'))
            config['dgp'].update(num_sites=2,site_min_samples=40,site_max_samples=40,covariate_dim=5,num_test=30)
            config['fisher'].update(local_epochs=1,batch_size=16)
            config['legacy_diagnostic'].update(iterations=2,evaluate_every=1)
            result, arrays, payloads = original_runner.compute_cell(config)
            sources[role] = dict(config=config,result=result,arrays=arrays,payloads=payloads,
                models={name:model_io.restore_model(record,arrays) for name,record in result['models'].items()})
        config = protocol.run_config(protocol.PILOT_TASK,protocol.cohort_plan()[0])
        config['sample_rows_per_site'] = 2
        binding = {key:config[key] for key in ('sources','accepted_report','original_source_provenance',
                                              'original_generation_source_files')}
        with mock.patch.object(runner,'load_sources',return_value=sources), \
             mock.patch.object(runner,'verify_source_bindings',return_value=binding), \
             mock.patch.object(fedfisher_em,'fit_local',side_effect=AssertionError('replay trained')), \
             mock.patch.object(fedfisher_em,'mean_categorical_fisher_diag',side_effect=AssertionError('replay recomputed Fisher')), \
             mock.patch.object(fedfisher_em,'legacy_validation_diagnostic',side_effect=AssertionError('replay selected a model')):
            replay = runner.compute_replay(config)
        self.assertTrue(replay['data_replay']['passed'])
        self.assertTrue(replay['diagnostic']['implementation_gate']['passed'])
        self.assertTrue(replay['diagnostic']['full_H_endpoint']['replay_matches_source'])
        self.assertIsNotNone(replay['diagnostic']['model_quality'])
        identity = dict(run_id=config['run_id'],seed=0,recipe=config['recipe'],dataset=config['dataset'],
                        config_sha256='a'*64,provenance_sha256='b'*64)
        replay['identity'] = identity
        with mock.patch.object(validation,'load_sources',return_value=sources), \
             mock.patch.object(protocol,'run_config',return_value=config), \
             mock.patch('torch.nn.Module._call_impl',side_effect=AssertionError('validator performed inference')), \
             mock.patch.object(original_runner,'generate_data',side_effect=AssertionError('validator regenerated data')):
            validation.validate_result(replay,config,identity)
            for mutation in ('expansion','data','timing'):
                bad = copy.deepcopy(replay)
                if mutation=='expansion': bad['expansion_authorized']=True
                elif mutation=='data': bad['data_replay']['test_raw_x_sha256']='0'*64
                else: bad['timing']['stages']['train']['calls']=1
                with self.assertRaises(ValueError): validation.validate_result(bad,config,identity)


if __name__ == '__main__':
    unittest.main()
