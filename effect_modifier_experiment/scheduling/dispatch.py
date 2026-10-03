"""Load a scientific runner only after the allocated runtime is configured."""
from importlib import import_module

from .catalog import TASK_CONTRACTS
from .errors import AdapterError
from .runtime import configure_runtime


def execute_task(job, runtime=configure_runtime):
    runtime(job['job_id'], job['allocation']['cpu_cores'])
    contract = TASK_CONTRACTS.get(job['task'])
    if contract is None:
        raise AdapterError('unreachable task dispatch')
    runner = getattr(import_module(contract.runner_module), contract.runner_name)
    return runner(job)
