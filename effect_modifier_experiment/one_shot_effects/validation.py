"""Read-only scientific completion checks using saved summaries and evidence.

No refitting, dataset generation, model inference, or hospital callbacks occur
here. Artifact hashes prove integrity; summary arithmetic checks fit semantics.
Raw-row metrics remain simulation observations, not cryptographic proof of a run.
"""
import hashlib
import re

import numpy as np

from . import protocol
from .evaluation import recovery,score_agreement,summarize_linearization
from .model_io import restore_model,model_wire
from .wire import canonical,communication_report,unpack_payload,unpad_payload


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _same(actual,expected,message):
    _require(canonical(actual)==canonical(expected),message)


def _close(actual,expected,message,atol=1e-10):
    a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    _require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
             and np.allclose(a,b,rtol=1e-9,atol=atol),message)


def _hash(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _nonnegative(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and np.isfinite(value) and value>=0


def _timing(record,recipe):
    _require(record['method']==recipe and record['actual_network_measured'] is False
             and record['execution_model']=='single_process_simulation','timing scope mismatch')
    stages=record['stages']
    _require(set(stages)=={'data_preparation','train','tuning','aggregation','evaluation'},'timing stage inventory mismatch')
    for name,stage in stages.items():
        for bucket in [stage,*stage['components'].values()]:
            _require(all(_nonnegative(bucket[k]) for k in ('wall_seconds','cpu_seconds'))
                and type(bucket['calls']) is int and bucket['calls']>=0
                and bucket['failed_calls']==0 and bucket['measured']==(bucket['calls']>0),
                'invalid stage measurement')
        if name!='tuning':
            _require(stage['measured'] and stage['calls']>0,'required cost component unmeasured')
    for key in ('wall_seconds','cpu_seconds'):
        _close(record['measured_stage_sum'][key],sum(s[key] for s in stages.values()),'timing totals mismatch')


def validate_result(result,arrays,payloads,config,identity):
    canonical(result)  # Refuse nonfinite JSON, unserializable evaluator objects.
    fixed=protocol.run_config(config['task'],dict(recipe=config['recipe'],dataset=config['dataset']))
    _same(config,fixed,'configuration is not the reviewed fixed pilot recipe')
    _same(result['identity'],identity,'scientific identity mismatch')
    _require(result['status']=='complete' and len(payloads)==config['dgp']['num_sites'],
             'incomplete result or site count mismatch')
    definition=config['recipe_definition']
    family,d=definition['family'],config['dgp']['covariate_dim']
    evidence=result['data_fingerprints']
    sites=evidence['sites']
    _require(len(sites)==len(payloads) and evidence['test_rows']==config['dgp']['num_test']
             and _hash(evidence['test']),'dataset evidence mismatch')
    for i,site in enumerate(sites):
        n=site['n_total']
        _require(site['site_id']==i and type(n) is int
            and config['dgp']['site_min_samples']<=n<=config['dgp']['site_max_samples'],
            'site sample count or identity mismatch')
        _same(site['counts'],dict(train=70*n//100,validation=15*n//100,
                                  holdout=n-70*n//100-15*n//100),
              'independent 70/15/15 split counts mismatch')
        _require(all(_hash(site[k]) for k in ('train','validation','holdout','partition')), 'missing split fingerprints')
    counts=np.asarray([s['n_total'] for s in sites],dtype=float)
    weights=counts/counts.sum()
    _close(evidence['target_weights'],weights,'target weights do not use original site counts',atol=1e-15)
    mask=np.zeros(d,dtype=bool)
    if config['dataset_definition'].get('scenario')!='null':
        mask[:config['dgp'].get('signal_dim',2)]=True
    models=result['models']
    expected={'primary'}|({'matched_fedavg'} if family=='aggregation' else set())
    if family=='aggregation':
        legacy=result['diagnostics']['legacy_rule']
        _require(legacy['status'] in ('complete','failed'),'invalid legacy diagnostic state')
        if legacy['status']=='complete':
            expected|={'legacy_validation_selected','strict_legacy_uploaded_objective'}
        else:
            _require(bool(legacy['message']) and legacy['error_type'] in ('FloatingPointError','RuntimeError'),
                     'unexplained legacy diagnostic failure')
    _require(set(models)==expected,'model inventory differs from recipe')
    keys=[k for record in models.values() for k in record['array_keys'].values()]
    _require(len(keys)==len(set(keys)) and set(keys)==set(arrays),'model archive inventory mismatch')
    restored={name:restore_model(record,arrays) for name,record in models.items()}
    expected_kind='dr_sieve' if family=='dr_sieve' else 'r_common' if family=='r_common' else 'outcome'
    _require(models['primary']['kind']==expected_kind,'wrong primary method kind')
    holdout_hashes=None
    test_hashes=None
    for name,record in models.items():
        test,holdout=record['test'],record['holdout']
        if record['kind']=='outcome':
            _same(record['metadata'],dict(model_name=definition['model'],covariate_dim=d),'outcome architecture mismatch')
        elif record['kind']=='dr_sieve':
            _same(record['metadata'],dict(basis_config=dict(kind=definition['basis'],covariate_dim=d)),
                  'shared basis configuration mismatch')
        _require(test['score_definition']=='external_mean_absolute_tau_change'
            and holdout['score_definition']=='target_weighted_H_half_mean_squared_tau_change'
            and test['prediction_clipped'] is False,'score or prediction definition mismatch')
        for scores in (test['scores'],holdout['exact_squared_scores']):
            _require(np.asarray(scores).shape==(d,) and all(_nonnegative(v) for v in scores),'invalid feature scores')
        _same(test['recovery'],recovery(test['scores'],mask),'external tie-aware recovery mismatch')
        _same(holdout['recovery'],recovery(holdout['exact_squared_scores'],mask),'holdout tie-aware recovery mismatch')
        _close(holdout['target_weights'],weights,'holdout weights mismatch',atol=1e-15)
        _same(holdout['rows'],[s['counts']['holdout'] for s in sites],'holdout sample counts mismatch')
        local_scores=np.asarray(holdout['local_mean_scores'])
        _require(local_scores.shape==(len(sites),d) and np.isfinite(local_scores).all()
                 and np.min(local_scores)>=0,'invalid site mean scores')
        _close(holdout['exact_squared_scores'],weights@local_scores,'holdout site means weighted incorrectly')
        if holdout_hashes is None:
            holdout_hashes=holdout['permutation_sha256']
            test_hashes=(test['raw_x_sha256'],test['permutation_sha256'])
        _same(holdout['permutation_sha256'],holdout_hashes,'models used different H perturbations')
        _same((test['raw_x_sha256'],test['permutation_sha256']),test_hashes,'models used different test rows or perturbations')
        _require(len(holdout_hashes)==len(sites) and all(_hash(h) for h in holdout_hashes)
                 and _hash(test['raw_x_sha256']) and all(_hash(h) for h in test['permutation_sha256']),
                 'invalid evaluation fingerprints')
        scm=config['dataset_definition']['family']=='scmv2'
        _require(test['true_cate_available']==scm and test['test_rows']==evidence['test_rows']
            and test['sqrt_pehe']==test['cate_rmse']
            and (_nonnegative(test['cate_rmse']) if scm else test['cate_rmse'] is None)
            and _nonnegative(test['estimated_heterogeneity'])
            and _nonnegative(test['effect_out_of_bounds_fraction']) and test['effect_out_of_bounds_fraction']<=1+1e-14,
            'invalid CATE, heterogeneity, or clipping metrics')
        _require(test['perturbation_scope']==('within_external_test_site_target_pi' if scm else 'legacy_pooled_common_external_test')
            and len(test['permutation_sha256'])==(len(sites) if scm else 1),'external permutation scope mismatch')
    primary=models['primary']
    summary=primary['holdout'].get('summary_squared_scores')
    requires_summary=family in ('dr_sieve','r_common','compression')
    _require((summary is not None)==requires_summary,'summary score scope mismatch')
    if summary is not None:
        _same(primary['holdout']['summary_comparison'],
              score_agreement(primary['holdout']['exact_squared_scores'],summary),'score agreement mismatch')
    for site,blob,holdout_hash in zip(sites,payloads,holdout_hashes):
        _,meta=unpack_payload(blob)
        _require(meta['site_id']==site['site_id'] and meta['n_total']==site['n_total']
                 and meta['n_train']==site['counts']['train'],'upload site population mismatch')
        if requires_summary:
            _require(meta['n_holdout']==site['counts']['holdout'] and
                meta.get('perturbations_sha256',meta.get('permutation_sha256'))==holdout_hash,
                'uploaded summary uses different holdout counts or perturbations')
    from .fedfisher_em import FedFisherEMConfig,validate_server_result as validate_em
    fisher_config=FedFisherEMConfig(**config['fisher'])
    if family in ('aggregation','linearity','compression'):
        validate_em(payloads,restored['primary'],summary,primary['fit'],fisher_config,training_only=family!='compression')
    elif family=='dr_sieve':
        from .dr_sieve import DRSieveConfig,validate_server_result
        validate_server_result(payloads,restored['primary'],summary,primary['fit'],DRSieveConfig(**config['dr_sieve']))
    elif family=='r_common':
        from .r_baseline import RlearnerConfig,validate_server_result
        validate_server_result(payloads,restored['primary'],summary,primary['fit'],definition['penalty'],RlearnerConfig(**config['r_baseline']))
    else:
        decoded=[unpack_payload(p,'fedavg_common_v1') for p in payloads]
        for a,m in decoded:
            _same(m['config'],config['fisher'],'FedAvg local configuration mismatch')
            _require(set(a)=={'theta'} and m['model_name']==definition['model'] and m['covariate_dim']==d,
                     'FedAvg upload model mismatch')
        theta=weights@np.stack([a['theta'] for a,_ in decoded])
        _close(arrays[primary['array_keys']['theta']],theta,'FedAvg parameter mean mismatch')
        _same(primary['fit']['communication'],communication_report(payloads),'FedAvg bytes mismatch')
        _close(primary['fit']['site_weights'],weights,'FedAvg weights mismatch',atol=1e-15)
        _require(primary['fit']['completion_eligible'] is True and primary['fit']['fisher_computed'] is False,
                 'FedAvg fit evidence mismatch')
    if family in ('dr_sieve','r_common'):
        _close(summary,primary['holdout']['exact_squared_scores'],'exact basis summary score disagrees with H reference',atol=1e-9)
    if family=='aggregation':
        decoded=[unpack_payload(p) for p in payloads]
        _close(arrays[models['matched_fedavg']['array_keys']['theta']],
               weights@np.stack([a['theta'] for a,_ in decoded]),'matched FedAvg comparison changed local fits')
        if result['diagnostics']['legacy_rule']['status']=='complete':
            legacy=result['diagnostics']['legacy_rule']['report']
            _same(legacy['legacy_configuration'],config['legacy_diagnostic'],'legacy rule configuration mismatch')
            _require(legacy['historical_equivalence_claim'] is False and legacy['selection_weighting']=='target_pi_mean',
                     'legacy replay scope mismatch')
            from .fedfisher_em import validate_legacy_diagnostic,LegacyDiagnosticConfig
            validate_legacy_diagnostic(payloads,restored['legacy_validation_selected'],
                restored['strict_legacy_uploaded_objective'],legacy,fisher_config,
                LegacyDiagnosticConfig(**config['legacy_diagnostic']))
            _same(legacy['validation_site_samples'],[s['counts']['validation'] for s in sites],
                  'legacy evaluator used different V sample counts')
    if family in ('linearity','compression'):
        local=result['diagnostics']['local_linearization']
        _require(len(local)==len(sites) and all(v['full_gradient_directional_derivatives'] is True
            and v['parameter_square_matrix_allocated'] is False for v in local),'invalid linearization diagnostic scope')
        combined=summarize_linearization(local,weights,mask,config['linearity_gate'])
        _same(result['diagnostics']['linearization'],combined,'linearization evidence or gate was changed')
        _close(combined['scores']['exact'],primary['holdout']['exact_squared_scores'],'linearization used a different nonlinear reference')
        if family=='compression':
            _close(combined['scores']['low_rank'],summary,'compressed summary disagrees with direct surrogate')
    prerequisite=result.get('prerequisite')
    if family=='compression':
        _require(isinstance(prerequisite,dict) and _hash(prerequisite['result_sha256'])
            and prerequisite['gate']['compression_eligible'] is True
            and prerequisite['identity']['provenance_sha256']==identity['provenance_sha256']
            and prerequisite['identity']['run_id']==protocol.linearity_prerequisite(config)['run_id'],
            'missing accepted source-matched full-gradient prerequisite')
        from .artifacts import current_provenance
        from .prerequisites import compression_prerequisite
        _same(prerequisite,compression_prerequisite(config,current_provenance()),
              'saved expansion evidence no longer matches the validated full-gradient artifact')
    else:
        _require(prerequisite is None,'unexpected prerequisite')
    communication=result['communication']
    p=primary
    broadcast=model_wire(p['kind'],p['metadata'],{k:arrays[v] for k,v in p['array_keys'].items()})
    protocol_blob=canonical(config).encode()
    expected_communication=communication_report(payloads,len(protocol_blob)*len(sites),len(broadcast)*len(sites))
    for k,v in expected_communication.items():
        _same(communication[k],v,'actual wire communication mismatch: '+k)
    budget=protocol.compression_wire_budget(config)
    native=sum(len(unpad_payload(blob)) if budget else len(blob) for blob in payloads)
    _require(communication['native_upload_bytes']==native
        and communication['padding_bytes']==sum(map(len,payloads))-native
        and communication['wire_budget_per_site']==budget
        and (budget is None or all(len(blob)==budget for blob in payloads))
        and communication['protocol_sha256']==hashlib.sha256(protocol_blob).hexdigest()
        and communication['final_model_sha256']==hashlib.sha256(broadcast).hexdigest(),
        'broadcast or compression byte-budget evidence mismatch')
    _require(len(result['local_audits'])==len(sites),'missing local compute audits')
    from .audit_validation import validate_local_audits
    validate_local_audits(result,payloads,config)
    _timing(result['timing'],config['recipe'])
