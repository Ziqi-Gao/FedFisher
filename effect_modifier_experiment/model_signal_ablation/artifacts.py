"""Reuse immutable lifecycle and the existing family-specific semantic checks."""

from ..fedfisher_timing import artifacts as timing
from ..fedfisher_timing.validation import validate_result as validate_fisher_result
from ..causal_comparison.artifacts import validate_result as validate_comparison_result
from . import protocol

current_provenance = timing.current_provenance
MEMORY_SCOPE = 'Linux current-process lifetime high-water RSS; not cgroup peak or reservation'


def validate_execution(execution):
    timing.validate_execution(execution, job_prefix='fedfisher-ms-', max_suffix=115)


def validate_result(result, arrays, identity, config):
    validator = (validate_fisher_result if config['method']['family'] == 'legacy_fedfisher'
                 else validate_comparison_result)
    validator(result, arrays, identity, config)
    memory = result.get('memory')
    if (not isinstance(memory, dict) or set(memory) != {'process_peak_rss_kib', 'scope'}
            or type(memory['process_peak_rss_kib']) is not int or memory['process_peak_rss_kib'] <= 0
            or memory['scope'] != MEMORY_SCOPE):
        raise ValueError('missing model/signal process memory observation')


class ArtifactStore(timing.ArtifactStore):
    def _validate_config(self):
        parameters = {key: self.config[key] for key in ('setting', 'cell', 'variant')}
        parameters.update(shard_index=0,
            shard_count=1 if self.config['task'] == protocol.PILOT_TASK else 8)
        if timing.canonical(self.config) != timing.canonical(protocol.run_config(self.config['task'], parameters)):
            raise ValueError('model/signal configuration is not a reviewed fixed recipe')

    def _validate_execution(self, execution):
        validate_execution(execution)

    def _validate_result(self, result, arrays, identity):
        validate_result(result, arrays, identity, self.config)
