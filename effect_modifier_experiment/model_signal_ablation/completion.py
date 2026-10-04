"""New-source completion of exactly the sealed failed shards, using unchanged fits."""
from . import completion_protocol as protocol
from .recovery import RecoveryStore, execute_shard
from ..infrastructure.artifacts import canonical


class CompletionStore(RecoveryStore):
    execution_protocol = protocol
    source_closeout = protocol.SOURCE_CLOSEOUT

    def _validate_config(self):
        parameters = protocol.parameters_for_config(self.config)
        if canonical(self.config) != canonical(protocol.run_config(self.config['task'], parameters)):
            raise ValueError('completion configuration differs from its fixed scientific recipe')

    def identity(self, seed, split):
        identity = super().identity(seed, split)
        parameters = (protocol.PILOT_PARAMETERS if self.config['task'] == protocol.PILOT_TASK
                      else protocol.FORMAL_PARAMETERS)
        allowed = {value for p in parameters
                   if all(p[key] == self.config[key] for key in ('setting', 'cell', 'variant'))
                   for value in protocol.seed_sequence(self.config['task'], p)}
        if seed not in allowed:
            raise ValueError('scientific unit is outside the fixed completion namespace')
        return identity


def run_shard(job):
    from . import completion_acceptance as gate
    return execute_shard(job, protocol, gate, CompletionStore)
