"""Read-only full-gradient prerequisite evidence for compression recipes."""
import hashlib
from . import protocol
from .artifacts import EffectArtifactStore, read_json


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
