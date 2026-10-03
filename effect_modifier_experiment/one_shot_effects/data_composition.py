"""Compose observed data and its evidence without importing a runner."""
from .data import SCMConfig, LegacyConfig, generate_scm, generate_legacy, rows_fingerprint, partition_fingerprint


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
