"""Small predictor archive contract, separate from fitting and evaluator rows."""
from dataclasses import asdict
import hashlib

import numpy as np

from .wire import pack_payload


def encode_model(model):
    from .fisher_models import OutcomeEffectModel, model_to_arrays
    from .dr_sieve import DRSieveModel
    from ..causal_comparison.rlearner import LinearProbabilityEffectModel
    if isinstance(model,OutcomeEffectModel):
        return 'outcome',dict(model_name=model.model_name,covariate_dim=model.covariate_dim),model_to_arrays(model)
    if isinstance(model,DRSieveModel):
        return 'dr_sieve',dict(basis_config=asdict(model.basis_config)),{'beta':model.beta}
    if isinstance(model,LinearProbabilityEffectModel):
        return 'r_common',dict(covariate_dim=len(model.coefficients)-1),{'coefficients':model.coefficients}
    raise TypeError('unsupported probability effect model')


def model_wire(kind,metadata,arrays):
    return pack_payload('one_shot_final_model_'+kind,arrays,metadata)


def archive_model(model,name,archive):
    kind,metadata,arrays=encode_model(model)
    keys={k:name+'.'+k for k in arrays}
    for k,value in arrays.items():
        if keys[k] in archive:
            raise ValueError('duplicate model archive key')
        archive[keys[k]]=np.array(value,copy=True)
    return dict(kind=kind,metadata=metadata,array_keys=keys,
                model_sha256=hashlib.sha256(model_wire(kind,metadata,arrays)).hexdigest())


def restore_model(record,archive):
    from .fisher_models import restore_model as restore_outcome
    from .dr_sieve import DRSieveModel
    from .basis import BasisConfig
    from ..causal_comparison.rlearner import LinearProbabilityEffectModel
    arrays={name:archive[key] for name,key in record['array_keys'].items()}
    kind,metadata=record['kind'],record['metadata']
    if hashlib.sha256(model_wire(kind,metadata,arrays)).hexdigest()!=record['model_sha256']:
        raise ValueError('model semantic fingerprint mismatch')
    if kind=='outcome':
        return restore_outcome(arrays,metadata)
    if kind=='dr_sieve' and set(arrays)=={'beta'}:
        return DRSieveModel(arrays['beta'],BasisConfig(**metadata['basis_config']))
    if kind=='r_common' and set(arrays)=={'coefficients'}:
        if arrays['coefficients'].shape!=(metadata['covariate_dim']+1,):
            raise ValueError('R model dimension mismatch')
        return LinearProbabilityEffectModel(arrays['coefficients'])
    raise ValueError('invalid serialized effect model')
