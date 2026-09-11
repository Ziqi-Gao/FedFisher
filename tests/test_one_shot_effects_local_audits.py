"""Tiny local-report fixtures for pure read-only completion audit checks."""
import copy
from dataclasses import asdict, replace
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.one_shot_effects import (
    audit_validation, dr_nuisance, dr_sieve, fedavg, fedfisher_em, protocol, r_baseline,
)
from effect_modifier_experiment.one_shot_effects.basis import BasisConfig
from effect_modifier_experiment.one_shot_effects.fisher_models import array_hash as fisher_hash
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations
from effect_modifier_experiment.one_shot_effects.types import Rows, SitePartitions
from effect_modifier_experiment.one_shot_effects.wire import pack_payload, unpack_payload
from effect_modifier_experiment.causal_comparison.rlearner import RlearnerConfig


def observed_sites():
    sites = []
    for site_id, n in enumerate((44, 64)):
        rng = np.random.default_rng(15+site_id)
        x = rng.normal(size=(n, 5))
        a = (np.arange(n)%2).astype(float)
        y = ((np.arange(n)//2)%2).astype(float)
        ids = np.arange(n, dtype=np.int64)+site_id*1000
        t, v = 70*n//100, 15*n//100
        parts = [Rows(x[sl], a[sl], y[sl], ids[sl])
                 for sl in (slice(0,t), slice(t,t+v), slice(t+v,n))]
        sites.append(SitePartitions(site_id,n,*parts))
    return sites


def evidence(sites):
    return {"sites": [dict(site_id=s.site_id, n_total=s.n_total,
        counts={name:len(getattr(s,name)) for name in ("train","validation","holdout")},
        array_fingerprints={name:audit_validation.observed_array_fingerprints(getattr(s,name))
                            for name in ("train","validation","holdout")}) for s in sites]}


def local_fixture(recipe, sites=None):
    sites = observed_sites() if sites is None else sites
    config = protocol.run_config(protocol.PILOT_TASK, dict(recipe=recipe, dataset="scmv2_base"))
    config["dgp"].update(num_sites=len(sites),covariate_dim=5,site_min_samples=44,site_max_samples=64)
    dc = dr_nuisance.DRNuisanceConfig(crossfit_folds=2,nuisance_cv_folds=2,nuisance_lambda_count=2,
                                      nuisance_lambda_max=.3,nuisance_lambda_min_ratio=.5)
    rc = RlearnerConfig(crossfit_folds=2,nuisance_cv_folds=2,nuisance_lambda_count=2,
                        nuisance_lambda_max=.3,nuisance_lambda_min_ratio=.5)
    fc = replace(fedfisher_em.FedFisherEMConfig(),local_epochs=1,batch_size=16,basis_max_rows=2)
    config.update(dr_nuisance=asdict(dc),r_baseline=asdict(rc),fisher=asdict(fc))
    payloads, audits = [], []
    family = config["recipe_definition"]["family"]
    for site in sites:
        plan = make_permutations(len(site.holdout),5,70+site.site_id)
        if family == "dr_sieve":
            payload,audit = dr_sieve.build_local_payload(site,BasisConfig("linear",5),dc,
                dr_sieve.DRSieveConfig(**config["dr_sieve"]),0,plan)
        elif family == "r_common":
            payload,audit = r_baseline.build_local_payload(site,rc,0,plan)
        elif family == "fedavg":
            payload,audit = fedavg.build_local_payload(site,"SyntheticMLP",0,fc)
        else:
            local = fedfisher_em.fit_local(site.train,"SyntheticMLP",0,fc,site_id=site.site_id)
            payload = (fedfisher_em.build_payload(local,site.holdout.x,plan,fc,site.n_total,site.site_id)
                       if family == "compression" else fedfisher_em.build_training_payload(local,site.n_total,site.site_id))
            audit = local.report
        payloads.append(payload)
        audits.append(audit)
    return dict(data_fingerprints=evidence(sites),local_audits=audits),payloads,config


class LocalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = {recipe:local_fixture(recipe) for recipe in (
            "dr_linear","r_unpenalized","fedavg_shallow","aggregation_shallow",
            "compression_shallow_fisher_gradient_r16")}

    def test_all_family_reports_validate_without_any_training_or_inference(self):
        with mock.patch.object(dr_nuisance,"fit_dr_pseudo_outcomes",side_effect=AssertionError("fit forbidden")), \
             mock.patch.object(r_baseline,"crossfit_site_nuisances",side_effect=AssertionError("fit forbidden")), \
             mock.patch.object(fedfisher_em,"fit_local",side_effect=AssertionError("fit forbidden")):
            for result,payloads,config in self.fixtures.values():
                audit_validation.validate_local_audits(result,payloads,config)

    def test_both_hash_schemes_match_real_local_report_functions(self):
        rows = observed_sites()[0].train
        actual = audit_validation.observed_array_fingerprints(rows)
        self.assertEqual(actual["nuisance_arrays"]["row_ids"],dr_nuisance.array_hash(rows.row_ids))
        self.assertEqual(actual["nuisance_arrays"]["raw_x"],dr_nuisance.array_hash(rows.x))
        self.assertEqual(actual["fisher_arrays"]["row_ids"],fisher_hash(rows.row_ids))
        self.assertEqual(actual["fisher_arrays"]["raw_x"],fisher_hash(rows.x))
        self.assertNotEqual(actual["fisher_arrays"]["row_ids"],actual["nuisance_arrays"]["row_ids"])

    def test_dr_payload_nuisance_config_cannot_drift_from_recipe(self):
        result,payloads,config = self.fixtures["dr_linear"]
        changed=[]
        for payload in payloads:
            arrays,meta=unpack_payload(payload)
            meta["nuisance_config"]["crossfit_folds"] += 1
            changed.append(pack_payload("dr_sieve",arrays,meta))
        with self.assertRaisesRegex(ValueError,"nuisance config"):
            audit_validation.validate_local_audits(result,changed,config)

    def test_dr_cv_failures_wrong_seeds_counts_and_ids_are_rejected(self):
        original,payloads,config=self.fixtures["dr_linear"]
        def seed(a): a["nuisance"].__setitem__("seed",1)
        def converged(a): a["nuisance"]["crossfit_folds"][0]["models"]["mu0"].__setitem__("converged",False)
        def folds(a): a["nuisance"]["crossfit_folds"].pop()
        def clip(a): a["nuisance"]["crossfit_folds"][0].__setitem__("propensity_clip_count",999)
        def count(a): a["nuisance"]["crossfit_folds"][0]["treatment_arm_counts"].__setitem__(0,0)
        def ids(a): a.__setitem__("holdout_row_ids_sha256","0"*64)
        for mutate in (seed,converged,folds,clip,count,ids):
            result=copy.deepcopy(original)
            mutate(result["local_audits"][0])
            with self.assertRaises(ValueError):
                audit_validation.validate_local_audits(result,payloads,config)

    def test_allowed_constant_outcome_probability_and_disallowed_constant_propensity(self):
        site=observed_sites()[0]
        site=replace(site,train=Rows(site.train.x,site.train.a,np.zeros(len(site.train)),site.train.row_ids))
        result,payloads,config=local_fixture("dr_linear",[site])
        audit_validation.validate_local_audits(result,payloads,config)
        changed=copy.deepcopy(result)
        changed["local_audits"][0]["nuisance"]["crossfit_folds"][0]["models"]["mu0"]["probability"] = .4
        with self.assertRaisesRegex(ValueError,"smoothed"):
            audit_validation.validate_local_audits(changed,payloads,config)
        changed=copy.deepcopy(result)
        models=changed["local_audits"][0]["nuisance"]["crossfit_folds"][0]["models"]
        models["propensity"]["fit_kind"]="train_only_smoothed_bernoulli_constant"
        with self.assertRaisesRegex(ValueError,"constant propensity"):
            audit_validation.validate_local_audits(changed,payloads,config)

    def test_r_convergence_local_precision_and_input_binding(self):
        original,payloads,config=self.fixtures["r_unpenalized"]
        mutations=(lambda a:a["nuisance"].__setitem__("crossfit_seed",10),
                   lambda a:a["nuisance"]["outer_folds"][0]["propensity"].__setitem__("converged",False),
                   lambda a:a["legacy_precision_summary"].__setitem__("residual_variance",0.),
                   lambda a:a.__setitem__("training_row_ids_sha256","0"*64))
        for change in mutations:
            result=copy.deepcopy(original)
            change(result["local_audits"][0])
            with self.assertRaises(ValueError):
                audit_validation.validate_local_audits(result,payloads,config)

    def test_em_and_fedavg_epochs_updates_seed_fisher_and_H_binding(self):
        for recipe in ("aggregation_shallow","fedavg_shallow","compression_shallow_fisher_gradient_r16"):
            original,payloads,config=self.fixtures[recipe]
            for key,value in (("local_epochs",30),("gradient_updates",999),("seed",1),("fisher_examples",999),
                              ("train_x_sha256","0"*64)):
                result=copy.deepcopy(original)
                result["local_audits"][0][key]=value
                with self.assertRaises(ValueError):
                    audit_validation.validate_local_audits(result,payloads,config)
        result,payloads,config=self.fixtures["compression_shallow_fisher_gradient_r16"]
        arrays,meta=unpack_payload(payloads[0])
        meta["basis"]["basis_training_rows"] += 1
        with self.assertRaises(ValueError):
            audit_validation.validate_local_audits(result,[pack_payload("fedfisher_em_v1",arrays,meta),payloads[1]],config)
        changed=copy.deepcopy(result)
        changed["data_fingerprints"]["sites"][0]["array_fingerprints"]["holdout"]["fisher_arrays"]["raw_x"]="0"*64
        with self.assertRaisesRegex(ValueError,"H raw-X"):
            audit_validation.validate_local_audits(changed,payloads,config)


if __name__ == "__main__":
    unittest.main()
