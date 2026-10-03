"""Small data-only wire, endpoint, and reviewed-recipe boundary fixtures."""
from tests.paths import PROJECT_ROOT
import json
from pathlib import Path
import unittest

import numpy as np

from effect_modifier_experiment.one_shot_effects import protocol,wire,evaluation
from effect_modifier_experiment.one_shot_effects.data import SCMConfig,generate_scm
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations


class LinearEffect:
    def predict_tau(self,x):
        return 2*x[:,0]+.1*x[:,1]


class WireTests(unittest.TestCase):
    def test_scalar_matrix_roundtrip_determinism_and_charge_actual_padding(self):
        arrays={'scalar':np.asarray(1.5),'matrix':np.arange(6,dtype=float).reshape(2,3)}
        blob=wire.pack_payload('fixture',arrays,{'site':2})
        self.assertEqual(blob,wire.pack_payload('fixture',dict(reversed(list(arrays.items()))),{'site':2}))
        result,meta=wire.unpack_payload(blob,'fixture')
        self.assertEqual(result['scalar'].shape,())
        self.assertFalse(result['matrix'].flags.writeable)
        np.testing.assert_array_equal(result['matrix'],arrays['matrix'])
        padded=wire.pad_payload(blob,len(blob)+128)
        self.assertEqual(wire.unpad_payload(padded),blob)
        self.assertEqual(wire.communication_report([padded],13,25)['upload_bytes'],len(blob)+128)
        np.testing.assert_array_equal(wire.unpack_payload(padded)[0]['matrix'],arrays['matrix'])

    def test_nondata_truncation_wrong_method_and_corrupt_padding_rejected(self):
        for arrays,meta in (({'a':np.array([object()])},{}),({'a':np.array([np.nan])},{}),
                            ({'a':np.array([1.])},{'hospital_callback':lambda:None})):
            with self.assertRaises((ValueError,TypeError)):
                wire.pack_payload('fixture',arrays,meta)
        blob=wire.pack_payload('fixture',{'a':np.array([1.])},{})
        for bad in (blob[:-1],blob+b'x',blob[:12]):
            with self.assertRaises(ValueError):
                wire.unpack_payload(bad)
        with self.assertRaises(ValueError):
            wire.unpack_payload(blob,'another')
        padded=wire.pad_payload(blob,len(blob)+40)
        with self.assertRaises(ValueError):
            wire.unpack_payload(padded[:-1]+b'1')


class EndpointTests(unittest.TestCase):
    def test_absolute_and_half_squared_are_distinct_and_direct(self):
        x=np.array([[0.,0.],[1.,0.],[3.,0.],[5.,0.]])
        plan=make_permutations(4,2,77)
        model=LinearEffect()
        absolute=evaluation.exact_scores(model,x,plan)
        squared=evaluation.exact_scores(model,x,plan,squared=True)
        expected=np.mean([np.mean((2*x[:,0]-2*x[p,0])**2)/2 for p in plan[0]])
        self.assertAlmostEqual(squared[0],expected)
        self.assertNotAlmostEqual(absolute[0],squared[0])
        self.assertEqual(squared[1],0)

    def test_null_ties_and_declared_gate_fail(self):
        null=evaluation.recovery(np.zeros(5),np.zeros(5,dtype=bool))
        self.assertFalse(null['applicable'])
        self.assertIsNone(null['auroc'])
        tied=evaluation.recovery(np.zeros(5),np.array([True,True,False,False,False]))
        self.assertAlmostEqual(tied['probability_all_signals_selected'],1/10)
        agreement=evaluation.score_agreement(np.zeros(5),np.zeros(5))
        self.assertIsNone(agreement['spearman_rank_correlation'])
        self.assertAlmostEqual(agreement['top2_set_agreement_probability'],1/10)
        gate={'denominator_floor':1e-8,'relative_tau_rmse_max':.25,'relative_score_linf_max':.25}
        fail=evaluation.linearity_gate(np.ones(3),np.zeros(3),np.ones(5),np.zeros(5),gate)
        self.assertFalse(fail['compression_eligible'])

    def test_external_site_permutations_and_target_weights(self):
        data=generate_scm(SCMConfig(num_sites=2,site_min_samples=20,site_max_samples=30,
                                   covariate_dim=5,num_test=60),0,'covariate_shift')
        config=protocol.run_config(protocol.PILOT_TASK,{'recipe':'dr_linear','dataset':'scmv2_base'})['evaluation']
        result=evaluation.external_test(LinearEffect(),data.evaluation,0,config)
        expected=np.zeros(5)
        for i,weight in enumerate(data.evaluation.target_weights):
            rows=data.evaluation.x[data.evaluation.site_ids==i]
            plan=make_permutations(len(rows),5,config['test_seed_offset']+1000003*i,5)
            expected+=weight*evaluation.exact_scores(LinearEffect(),rows,plan)
        np.testing.assert_allclose(result['scores'],expected,rtol=0,atol=1e-14)
        self.assertEqual(len(result['permutation_sha256']),2)
        self.assertEqual(result['cate_rmse'],result['sqrt_pehe'])


class ProtocolTests(unittest.TestCase):
    def test_all_identities_fit_artifact_contract_and_no_formal_surface(self):
        ids=set()
        for recipe in protocol.RECIPES:
            for dataset in protocol.DATASETS:
                params={'recipe':recipe,'dataset':dataset}
                run_id=protocol.run_id(protocol.PILOT_TASK,params)
                self.assertLessEqual(len(run_id),128)
                self.assertNotIn(run_id,ids)
                ids.add(run_id)
        with self.assertRaises(ValueError):
            protocol.validate_parameters('effect-modifier-one-shot-effects-shard',params)
        for extra in ({'cpu':1},{'seed':1},{'num_seeds':300}):
            with self.assertRaises(ValueError):
                protocol.validate_parameters(protocol.PILOT_TASK,dict(params,**extra))

    def test_three_strategies_have_identical_public_byte_budgets(self):
        for architecture in ('shallow','deep'):
            for rank in (16,32,64):
                budgets=[]
                for strategy in ('fisher_gradient','euclidean_gradient','random_orthogonal'):
                    c=protocol.run_config(protocol.PILOT_TASK,dict(
                        recipe=f'compression_{architecture}_{strategy}_r{rank}',dataset='scmv2_base'))
                    budgets.append(protocol.compression_wire_budget(c))
                    self.assertEqual(protocol.linearity_prerequisite(c)['recipe'],'linearity_'+architecture)
                self.assertEqual(len(set(budgets)),1)
                self.assertGreater(budgets[0],0)
        self.assertEqual(len(protocol.staged_pilot_plan()['stage2']),12)

    def test_reviewable_stage_plan_matches_the_only_protocol_definition(self):
        path=PROJECT_ROOT/'effect_modifier_experiment/server_scheduler/one-shot-effects-v1/staged-pilot-plan.json'
        value=json.loads(path.read_text())
        self.assertEqual(value.pop('protocol'),protocol.PROTOCOL)
        self.assertIn('not requests',value.pop('purpose'))
        self.assertEqual(value,protocol.staged_pilot_plan())


if __name__=='__main__':
    unittest.main()
