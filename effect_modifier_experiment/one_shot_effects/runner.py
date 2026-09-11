"""Scientific composition, called only by the reviewed foreground adapter.

Local methods see observed partitions. Only the simulation evaluation phase
sees independent test rows/truth or performs exact post-aggregation H inference.
"""
import hashlib
import os

import numpy as np

from ..causal_comparison.measurement import MethodTimer
from . import protocol
from .artifacts import EffectArtifactStore, current_provenance, read_json
from .data import (SCMConfig,LegacyConfig,generate_scm,generate_legacy,
                   rows_fingerprint,partition_fingerprint)
from .evaluation import external_test,holdout_reference,score_agreement,summarize_linearization
from .interventions import make_permutations
from .model_io import archive_model,encode_model,model_wire
from .wire import canonical,communication_report,pad_payload,unpad_payload,unpack_payload


def generate_data(config):
    definition=config['dataset_definition']
    if definition['family']=='scmv2':
        return generate_scm(SCMConfig(**config['dgp']),config['seed'],definition['scenario'])
    return generate_legacy(LegacyConfig(**config['dgp']),config['seed'],definition['variant'],definition['split'])


def _data_evidence(data):
    from .audit_validation import observed_array_fingerprints
    return dict(sites=[dict(site_id=s.site_id,n_total=s.n_total,
        counts={key:len(getattr(s,key)) for key in ('train','validation','holdout')},
        partition=partition_fingerprint(s),
        array_fingerprints={key:observed_array_fingerprints(getattr(s,key))
                            for key in ('train','validation','holdout')},
        **{key:rows_fingerprint(getattr(s,key)) for key in ('train','validation','holdout')}) for s in data.sites],
        test=rows_fingerprint(data.evaluation.rows),test_rows=len(data.evaluation.rows),
        target_weights=data.evaluation.target_weights.tolist())


def compute_cell(config):
    """Bounded numerical fixtures may inject a tiny config; no path side effects.

    Production accepts only protocol.run_config through run_job. This function
    is not an alternative launcher and has no command-line entrypoint.
    """
    from . import fedfisher_em as em, fedavg, dr_sieve, r_baseline
    from .basis import BasisConfig
    from .dr_nuisance import DRNuisanceConfig
    from ..causal_comparison.rlearner import RlearnerConfig
    from .fisher_models import restore_model,model_to_arrays
    timer=MethodTimer(config['recipe'])
    definition=config['recipe_definition']
    family=definition['family']
    seed=config['seed']
    with timer.measure('data_preparation'):
        data=generate_data(config)
        sites=data.sites
        weights=data.evaluation.target_weights
        plans=[make_permutations(len(s.holdout),s.train.n_features,
            config['evaluation']['holdout_seed_offset']+seed+1000003*s.site_id,5) for s in sites]
        evidence=_data_evidence(data)
        protocol_blob=canonical(config).encode('utf-8')
    fisher_config=em.FedFisherEMConfig(**config['fisher'])
    payloads,local_audits,locals_=[] ,[],[]
    diagnostics={}
    budget=protocol.compression_wire_budget(config)
    with timer.measure('train'):
        for site,plan in zip(sites,plans):
            with timer.measure('train','site_%03d'%site.site_id):
                if family in ('aggregation','linearity','compression'):
                    local=em.fit_local(site.train,definition['model'],seed,fisher_config,site_id=site.site_id)
                    locals_.append(local)
                    blob=(em.build_payload(local,site.holdout.x,plan,fisher_config,
                            n_total=site.n_total,site_id=site.site_id) if family=='compression'
                          else em.build_training_payload(local,n_total=site.n_total,site_id=site.site_id))
                    audit=local.report
                elif family=='fedavg':
                    blob,audit=fedavg.build_local_payload(site,definition['model'],seed,fisher_config)
                elif family=='dr_sieve':
                    blob,audit=dr_sieve.build_local_payload(site,
                        BasisConfig(definition['basis'],site.train.n_features),
                        DRNuisanceConfig(**config['dr_nuisance']),dr_sieve.DRSieveConfig(**config['dr_sieve']),
                        seed,plan)
                elif family=='r_common':
                    blob,audit=r_baseline.build_local_payload(site,RlearnerConfig(**config['r_baseline']),seed,plan)
                else:
                    raise ValueError('unsupported scientific family')
                payloads.append(pad_payload(blob,budget) if budget else blob)
                local_audits.append(audit)
    summary_scores=None
    with timer.measure('aggregation'):
        if family in ('aggregation','linearity'):
            primary,fit=em.aggregate_training_payloads(payloads,fisher_config)
        elif family=='compression':
            primary,summary_scores,fit=em.aggregate(payloads,fisher_config)
        elif family=='fedavg':
            primary,fit=fedavg.aggregate(payloads,fisher_config)
        elif family=='dr_sieve':
            primary,summary_scores,fit=dr_sieve.aggregate(payloads,dr_sieve.DRSieveConfig(**config['dr_sieve']))
        else:
            primary,summary_scores,fit=r_baseline.aggregate(payloads,definition['penalty'],RlearnerConfig(**config['r_baseline']))
        kind,metadata,model_arrays=encode_model(primary)
        broadcast=model_wire(kind,metadata,model_arrays)
    models={'primary':(primary,fit)}
    # This evaluator-only replay is explicitly outside the one-upload server.
    # It reuses local T models; its own selection compute is measured separately.
    if family=='aggregation':
        with timer.measure('tuning'):
            with timer.measure('tuning','legacy_validation_rule_replay'):
                try:
                    selected,strict_old,legacy=em.legacy_validation_diagnostic(payloads,
                        [s.validation for s in sites],fisher_config,
                        legacy_config=em.LegacyDiagnosticConfig(**config['legacy_diagnostic']))
                    models['legacy_validation_selected']=(selected,{'scope':'simulation_diagnostic_only'})
                    models['strict_legacy_uploaded_objective']=(strict_old,legacy['strict_legacy_uploaded_objective'])
                    diagnostics['legacy_rule']=dict(status='complete',report=legacy)
                except (FloatingPointError,RuntimeError) as error:
                    # A historical undefined/unstable rule is a diagnostic
                    # finding, not permission to alter it or hide its failure.
                    diagnostics['legacy_rule']=dict(status='failed',error_type=type(error).__name__,message=str(error))
            with timer.measure('tuning','matched_local_FedAvg_reference'):
                theta=weights@np.stack([s.theta for s in locals_])
                matched=restore_model({'theta':theta},dict(model_name=definition['model'],covariate_dim=sites[0].train.n_features))
                models['matched_fedavg']=(matched,dict(scope='same_local_T_models_aggregation_reference',
                    cost_rule='reuses_EM_local_fits; standalone_fedavg_recipe_measures_own_training'))
    archive,records={},{}
    with timer.measure('evaluation'):
        truth=data.evaluation.truth
        mask=(truth.modifier_mask if truth is not None else
              np.arange(sites[0].train.n_features)<config['dgp'].get('signal_dim',2))
        for name,(model,report) in models.items():
            with timer.measure('evaluation',name):
                record=archive_model(model,name,archive)
                record.update(fit=report,test=external_test(model,data.evaluation,seed,config['evaluation']),
                              holdout=holdout_reference(model,sites,plans,mask))
                if name=='primary' and summary_scores is not None:
                    record['holdout'].update(summary_squared_scores=summary_scores.tolist(),
                        summary_comparison=score_agreement(record['holdout']['exact_squared_scores'],summary_scores))
                records[name]=record
        if family in ('linearity','compression'):
            with timer.measure('evaluation','local_linearization_diagnostics'):
                theta=model_to_arrays(primary)['theta']
                local_diagnostics=[(em.simulation_diagnostics(local,blob,theta,site.holdout.x,plan)
                    if family=='compression' else em.full_gradient_diagnostics(local,theta,site.holdout.x,plan))
                    for local,blob,site,plan in zip(locals_,payloads,sites,plans)]
                diagnostics['local_linearization']=local_diagnostics
                diagnostics['linearization']=summarize_linearization(local_diagnostics,weights,mask,config['linearity_gate'])
    native=sum(len(unpad_payload(blob)) if budget else len(blob) for blob in payloads)
    communication=communication_report(payloads,len(protocol_blob)*len(sites),len(broadcast)*len(sites))
    communication.update(native_upload_bytes=native,padding_bytes=sum(map(len,payloads))-native,
        wire_budget_per_site=budget,protocol_sha256=hashlib.sha256(protocol_blob).hexdigest(),
        final_model_sha256=hashlib.sha256(broadcast).hexdigest(),
        broadcasts='serialized common protocol and primary final model, separately charged per site')
    return dict(status='complete',data_fingerprints=evidence,models=records,communication=communication,
        timing=timer.to_dict(),local_audits=local_audits,diagnostics=diagnostics,
        scope='single_process_scientific_simulation; no_network_latency_measurement'),archive,payloads


def compression_prerequisite(config,provenance):
    prerequisite=protocol.linearity_prerequisite(config)
    if prerequisite is None:
        return None
    store=EffectArtifactStore(prerequisite,provenance)
    if store.status()!='validated':
        raise ValueError('matching full-gradient pilot must complete before any low-rank training')
    result=read_json(store.unit_dir/'result.json')
    if result['diagnostics']['linearization']['gate']['compression_eligible'] is not True:
        raise ValueError('full-gradient linearization gate failed; preserve and analyze it before expansion')
    return dict(identity=store.identity,
                result_sha256=hashlib.sha256((store.unit_dir/'result.json').read_bytes()).hexdigest(),
                gate=result['diagnostics']['linearization']['gate'])


def run_job(job):
    """One seed-zero cell, under validated scheduler allocation and provenance."""
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()!=1:
            raise
    config=protocol.run_config(job['task'],job['parameters'])
    provenance=current_provenance()
    prerequisite=compression_prerequisite(config,provenance)
    store=EffectArtifactStore(config,provenance)
    store.initialize()
    if store.status()=='validated':
        return 0
    execution={key:job[key] for key in ('job_id','execution_profile','allocation')}
    execution.update(attempt=int(os.environ['SERVER_SCHEDULER_ATTEMPT']),
                     estimated_runtime_seconds=job['estimated_runtime_seconds'])
    store.start(execution)
    try:
        result,arrays,payloads=compute_cell(config)
        result['identity']=store.identity
        result['prerequisite']=prerequisite
        store.complete(result,arrays,payloads)
    except Exception as error:
        try:
            store.fail(error)
        except Exception as evidence_error:
            raise RuntimeError('scientific attempt failed and failure evidence could not be published: '+str(evidence_error)) from error
        raise
    return 0
