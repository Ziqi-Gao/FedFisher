"""Reviewed task catalog; values here do not imply central registration.

Protocol and runner imports are lazy strings. Importing this catalog must never
load NumPy/PyTorch before the scheduler allocation sets the thread environment.
"""
from dataclasses import dataclass
from pathlib import Path
import re

PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")

SCRATCH_ROOT = Path("/scr/del6500/FedFisher")

PROJECT_NAME = "FedFisher"

SCHEMA_VERSION = 2

CPU_PROFILE = "cpu-1core"

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

VARIANTS = (
    "control",
    "site_gaussian_shift",
    "local_permutation",
    "site_gaussian_local_permutation",
)

DSLASSO_TASK = "effect-modifier-dslassocov-shard"

ONE_SHOT_TASK = "effect-modifier-one-shot-lasso-shard"

COMPARISON_PILOT_TASK = "effect-modifier-comparison-pilot"

COMPARISON_TASK = "effect-modifier-comparison-shard"

ONE_SHOT_EFFECTS_PILOT_TASK = "effect-modifier-one-shot-effects-pilot"

EFFECTS_REPLAY_PILOT_TASK = "effect-modifier-effects-replay-pilot"

FEDFISHER_TIMING_PILOT_TASK = "effect-modifier-fedfisher-timing-pilot"

FEDFISHER_TIMING_TASK = "effect-modifier-fedfisher-timing-shard"

MODEL_SIGNAL_PILOT_TASK = "effect-modifier-model-signal-pilot"

MODEL_SIGNAL_TASK = "effect-modifier-model-signal-shard"

MODEL_SIGNAL_RECOVERY_PILOT_TASK = "effect-modifier-model-signal-recovery-pilot"

MODEL_SIGNAL_RECOVERY_TASK = "effect-modifier-model-signal-recovery-shard"

MODEL_SIGNAL_COMPLETION_PILOT_TASK = "effect-modifier-model-signal-completion-pilot"

MODEL_SIGNAL_COMPLETION_TASK = "effect-modifier-model-signal-completion-shard"

SURVIVAL_PILOT_TASK = "effect-modifier-survival-pilot"

SURVIVAL_TASK = "effect-modifier-survival-shard"

SURVIVAL_TASKS = (SURVIVAL_PILOT_TASK, SURVIVAL_TASK)

SURVIVAL_POOLED_PILOT_TASK = "effect-modifier-survival-pooled-pilot"

SURVIVAL_POOLED_TASK = "effect-modifier-survival-pooled-shard"

SURVIVAL_POOLED_TASKS = (SURVIVAL_POOLED_PILOT_TASK, SURVIVAL_POOLED_TASK)

SURVIVAL_DIAGNOSTIC_TASK = "effect-modifier-survival-dslasso-diagnostic"

SURVIVAL_RECOVERY_TASK = "effect-modifier-survival-dslasso-recovery-shard"

SURVIVAL_REPAIR_PILOT_TASK = "effect-modifier-survival-dslasso-repair-pilot"

SURVIVAL_REPAIR_TASK = "effect-modifier-survival-dslasso-repair-shard"

SURVIVAL_INTEGRATED_PILOT_TASK = "effect-modifier-survival-dslasso-integrated-pilot"

SURVIVAL_INTEGRATED_TASK = "effect-modifier-survival-dslasso-integrated-shard"

SURVIVAL_RECOVERY_TASKS = (SURVIVAL_DIAGNOSTIC_TASK, SURVIVAL_RECOVERY_TASK,
                           SURVIVAL_REPAIR_PILOT_TASK, SURVIVAL_REPAIR_TASK,
                           SURVIVAL_INTEGRATED_PILOT_TASK, SURVIVAL_INTEGRATED_TASK)

COMPARISON_TASKS = (COMPARISON_PILOT_TASK, COMPARISON_TASK)

FEDFISHER_TIMING_TASKS = (FEDFISHER_TIMING_PILOT_TASK, FEDFISHER_TIMING_TASK)

MODEL_SIGNAL_TASKS = (MODEL_SIGNAL_PILOT_TASK, MODEL_SIGNAL_TASK)

MODEL_SIGNAL_RECOVERY_TASKS = (MODEL_SIGNAL_RECOVERY_PILOT_TASK, MODEL_SIGNAL_RECOVERY_TASK)

MODEL_SIGNAL_COMPLETION_TASKS = (MODEL_SIGNAL_COMPLETION_PILOT_TASK, MODEL_SIGNAL_COMPLETION_TASK)

JOB_FIELDS = frozenset(
    (
        "schema_version",
        "job_id",
        "project",
        "task",
        "resources",
        "allocation",
        "parameters",
        "priority",
        "submitted_at",
        "state",
        "updated_at",
        "requested_profile",
        "execution_profile",
        "estimated_runtime_seconds",
        "gpu_indices",
        "gpu_uuids",
        "gpu_pci_bus_ids",
        "cpu_ids",
        "numa_node",
        "stdout_log",
        "stderr_log",
        "exit_code",
        "failure_reason",
    )
)

RESOURCE_FIELDS = frozenset(
    (
        "cpu_cores",
        "memory_mib",
        "gpu_count",
        "gpu_memory_mib",
        "gpu_utilization_pct",
        "exclusive_gpu",
    )
)


@dataclass(frozen=True)
class TaskContract:
    protocol_module: str
    runner_module: str
    runner_name: str
    memory_mib: int = 8192


_BASE = 'effect_modifier_experiment.'
TASK_CONTRACTS = {
    DSLASSO_TASK: TaskContract('', _BASE + 'scheduling.legacy', '_run_dslasso', 4096),
    ONE_SHOT_TASK: TaskContract('', _BASE + 'scheduling.legacy', '_run_one_shot'),
}
for _names, _package, _runner, _entrypoint in (
    (COMPARISON_TASKS, 'causal_comparison', 'runner', 'run_shard'),
    ((ONE_SHOT_EFFECTS_PILOT_TASK,), 'one_shot_effects', 'runner', 'run_job'),
    ((EFFECTS_REPLAY_PILOT_TASK,), 'one_shot_effects.diagnostics', 'runner', 'run_job'),
    (FEDFISHER_TIMING_TASKS, 'fedfisher_timing', 'runner', 'run_shard'),
    (MODEL_SIGNAL_TASKS, 'model_signal_ablation', 'runner', 'run_shard'),
    (MODEL_SIGNAL_RECOVERY_TASKS, 'model_signal_ablation', 'recovery', 'run_shard'),
    (SURVIVAL_TASKS, 'survival', 'runner', 'run_job'),
    (SURVIVAL_POOLED_TASKS, 'survival_pooled', 'runner', 'run_job'),
    (SURVIVAL_RECOVERY_TASKS, 'survival', 'recovery', 'run_job'),
):
    _protocol = 'recovery_protocol' if _runner == 'recovery' else 'protocol'
    for _name in _names:
        if _name in TASK_CONTRACTS:
            raise ValueError('duplicate reviewed task: ' + _name)
        TASK_CONTRACTS[_name] = TaskContract(
            _BASE + _package + '.' + _protocol, _BASE + _package + '.' + _runner, _entrypoint)

for _name in MODEL_SIGNAL_COMPLETION_TASKS:
    if _name in TASK_CONTRACTS:
        raise ValueError('duplicate reviewed task: ' + _name)
    TASK_CONTRACTS[_name] = TaskContract(
        _BASE + 'model_signal_ablation.completion_protocol',
        _BASE + 'model_signal_ablation.completion', 'run_shard')

ALLOWED_TASKS = frozenset(TASK_CONTRACTS)
TASK_PROFILES = {name: dict(execution_profile=CPU_PROFILE, memory_mib=contract.memory_mib)
                 for name, contract in TASK_CONTRACTS.items()}


def adapter_review_sources():
    """Bind implementations as well as the stable external command path."""
    directory = PROJECT_ROOT / 'effect_modifier_experiment/scheduling'
    return {'adapter_' + name: directory / (name + '.py')
            for name in ('adapter', 'catalog', 'validation', 'runtime', 'dispatch', 'legacy', 'errors', '__init__')}
