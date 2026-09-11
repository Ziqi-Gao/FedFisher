"""Fixed-local-epoch FedAvg baseline on the new common T/V/H protocol."""
from dataclasses import asdict

import numpy as np

from .fedfisher_em import fit_local
from .fisher_models import restore_model
from .wire import communication_report, pack_payload, unpack_payload


def build_local_payload(site, model_name, seed, config):
    local=fit_local(site.train,model_name,seed,config,site_id=site.site_id,compute_fisher=False)
    metadata=dict(site_id=site.site_id,n_total=site.n_total,n_train=len(site.train),
                  model_name=model_name,covariate_dim=site.train.n_features,
                  config=asdict(config),validation_use='unused_fixed_local_epochs',
                  coordinates='shared_raw_A_X_AX')
    return pack_payload('fedavg_common_v1',{'theta':local.theta},metadata),local.report


def aggregate(payloads, config):
    sites=[unpack_payload(p,'fedavg_common_v1') for p in payloads]
    if not sites:
        raise ValueError('nonempty FedAvg payload list required')
    meta=sites[0][1]
    ids=[]
    for arrays,m in sites:
        if (set(arrays)!={'theta'} or m['config']!=asdict(config)
                or any(m[k]!=meta[k] for k in ('model_name','covariate_dim','coordinates'))
                or type(m['n_total']) is not int or m['n_total']<=0
                or type(m['site_id']) is not int or m['site_id']<0
                or arrays['theta'].shape!=sites[0][0]['theta'].shape):
            raise ValueError('incompatible common-protocol FedAvg payload')
        ids.append(m['site_id'])
    if len(set(ids))!=len(ids):
        raise ValueError('duplicate site')
    counts=np.asarray([m['n_total'] for _,m in sites],dtype=np.float64)
    weights=counts/counts.sum()
    theta=weights@np.stack([a['theta'] for a,_ in sites])
    model=restore_model({'theta':theta},meta)
    return model,dict(method='FedAvg_common_target_weights',completion_eligible=True,
        site_weights=weights.tolist(),fisher_computed=False,
        validation_use='unused_fixed_local_epochs',score_scope='independent_evaluator_only',
        communication=communication_report(payloads))
