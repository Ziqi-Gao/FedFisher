#!/scr/del6500/FedFisher/envs/fedfisher-py39/bin/python
"""Foreground ServerScheduler adapter for reviewed FedFisher CPU shards."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys


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
SURVIVAL_PILOT_TASK = "effect-modifier-survival-pilot"
SURVIVAL_TASK = "effect-modifier-survival-shard"
SURVIVAL_TASKS = (SURVIVAL_PILOT_TASK, SURVIVAL_TASK)
SURVIVAL_DIAGNOSTIC_TASK = "effect-modifier-survival-dslasso-diagnostic"
SURVIVAL_RECOVERY_TASK = "effect-modifier-survival-dslasso-recovery-shard"
SURVIVAL_REPAIR_PILOT_TASK = "effect-modifier-survival-dslasso-repair-pilot"
SURVIVAL_REPAIR_TASK = "effect-modifier-survival-dslasso-repair-shard"
SURVIVAL_RECOVERY_TASKS = (SURVIVAL_DIAGNOSTIC_TASK, SURVIVAL_RECOVERY_TASK,
                           SURVIVAL_REPAIR_PILOT_TASK, SURVIVAL_REPAIR_TASK)
# These new contracts are project-side proposals until centrally registered.
COMPARISON_TASKS = (COMPARISON_PILOT_TASK, COMPARISON_TASK)
FEDFISHER_TIMING_TASKS = (FEDFISHER_TIMING_PILOT_TASK, FEDFISHER_TIMING_TASK)
MODEL_SIGNAL_TASKS = (MODEL_SIGNAL_PILOT_TASK, MODEL_SIGNAL_TASK)
ALLOWED_TASKS = frozenset(
    (DSLASSO_TASK, ONE_SHOT_TASK, *COMPARISON_TASKS, ONE_SHOT_EFFECTS_PILOT_TASK,
     EFFECTS_REPLAY_PILOT_TASK, *FEDFISHER_TIMING_TASKS, *SURVIVAL_TASKS,
     *SURVIVAL_RECOVERY_TASKS, *MODEL_SIGNAL_TASKS)
)
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
TASK_PROFILES = {
    DSLASSO_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 4096},
    ONE_SHOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    COMPARISON_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    COMPARISON_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    ONE_SHOT_EFFECTS_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    EFFECTS_REPLAY_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    FEDFISHER_TIMING_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    FEDFISHER_TIMING_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    MODEL_SIGNAL_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    MODEL_SIGNAL_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    SURVIVAL_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    SURVIVAL_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    SURVIVAL_DIAGNOSTIC_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    SURVIVAL_RECOVERY_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    SURVIVAL_REPAIR_PILOT_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
    SURVIVAL_REPAIR_TASK: {"execution_profile": CPU_PROFILE, "memory_mib": 8192},
}
DSLASSO_RUN_IDS = {
    "control": "dslassocov_control_server_scheduler_v2r2_300seeds",
    "site_gaussian_shift": "dslassocov_site_gaussian_shift_server_scheduler_v2r2_300seeds",
    "local_permutation": "dslassocov_local_permutation_server_scheduler_v2r2_300seeds",
    "site_gaussian_local_permutation": (
        "dslassocov_site_gaussian_local_permutation_server_scheduler_v2r2_300seeds"
    ),
}
ONE_SHOT_RUN_SUFFIX = (
    "stationarity_4000iter_full_scope_server_scheduler_v2r2_300seeds"
)
DSLASSO_ENVIRONMENT = {
    "SEED_START": "0",
    "SEED_END": "299",
    "SPLITS": "iid,noniid",
    "SYNTHETIC_ALPHA": "0.1",
    "EFFECT_MODIFIER_NUM_SITES": "10",
    "SYNTHETIC_NUM_TEST": "10000",
    "EFFECT_MODIFIER_COVARIATE_DIM": "100",
    "EFFECT_MODIFIER_SIGNAL_DIM": "2",
    "EFFECT_MODIFIER_SIGNAL_STRENGTH": "2.0",
    "EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH": "1.0",
    "EFFECT_MODIFIER_CONFOUNDING_STRENGTH": "1.0",
    "EFFECT_MODIFIER_INTERCEPT": "0.0",
    "EFFECT_MODIFIER_TREATMENT_PROB": "0.5",
    "EFFECT_MODIFIER_PROPENSITY_CLIP": "0.05",
    "EFFECT_MODIFIER_SITE_MIN_SAMPLES": "500",
    "EFFECT_MODIFIER_SITE_MAX_SAMPLES": "1000",
    "EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN": "0.75",
    "EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW": "0.75",
    "EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH": "1.25",
    "PREDICTION_INTERVENTION_MODES": "permute,zero",
    "PREDICTION_INTERVENTION_REPEATS": "5",
    "PREDICTION_INTERVENTION_BATCH_SIZE": "1024",
    "CV_FOLDS": "5",
    "CV_SEED_OFFSET": "4000037",
    "LAMBDA_COUNT": "30",
    "LAMBDA_MIN_RATIO": "0.001",
    "STANDARDIZATION_SCALE_FLOOR": "1e-12",
    "OPTIMIZER_MAX_ITER": "2000",
    "OPTIMIZER_TOLERANCE": "1e-6",
    "OPTIMIZER_INITIAL_STEP": "1.0",
    "OPTIMIZER_BACKTRACK_FACTOR": "0.5",
    "OPTIMIZER_MIN_STEP": "1e-12",
    "THREADS_PER_WORKER": "1",
}


class AdapterError(ValueError):
    """A fail-closed job-contract or task-contract violation."""


def _only_keys(payload, expected, context):
    if not isinstance(payload, dict):
        raise AdapterError("%s must be an object" % context)
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        raise AdapterError(
            "%s fields mismatch; missing=%s unknown=%s"
            % (context, ",".join(missing), ",".join(unknown))
        )


def _integer(value, field, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterError("%s must be an integer" % field)
    if minimum is not None and value < minimum:
        raise AdapterError("%s must be >= %d" % (field, minimum))
    if maximum is not None and value > maximum:
        raise AdapterError("%s must be <= %d" % (field, maximum))
    return value


def _number(value, field, minimum=None, strict_minimum=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterError("%s must be numeric" % field)
    value = float(value)
    if not math.isfinite(value):
        raise AdapterError("%s must be finite" % field)
    if minimum is not None and (
        value <= minimum if strict_minimum else value < minimum
    ):
        operator = ">" if strict_minimum else ">="
        raise AdapterError("%s must be %s %s" % (field, operator, minimum))
    return value


def _environment_integer(environment, key, minimum=None):
    value = environment.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise AdapterError("%s must be an unsigned integer string" % key)
    return _integer(int(value), key, minimum)


def _environment_number(environment, key, minimum=None, strict_minimum=False):
    value = environment.get(key)
    if not isinstance(value, str):
        raise AdapterError("%s must be a numeric string" % key)
    try:
        numeric = float(value)
    except ValueError as error:
        raise AdapterError("%s must be a numeric string" % key) from error
    return _number(numeric, key, minimum, strict_minimum)


def _identifier(value, field):
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise AdapterError("%s is not a valid identifier" % field)
    return value


def _timestamp(value, field):
    if not isinstance(value, str):
        raise AdapterError("%s must be an ISO-8601 string" % field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AdapterError("%s must be an ISO-8601 string" % field) from error
    if parsed.tzinfo is None:
        raise AdapterError("%s must include a timezone" % field)


def _resource(payload, field):
    _only_keys(payload, RESOURCE_FIELDS, field)
    normalized = {
        "cpu_cores": _integer(payload["cpu_cores"], field + ".cpu_cores", 1),
        "memory_mib": _integer(payload["memory_mib"], field + ".memory_mib", 1),
        "gpu_count": _integer(payload["gpu_count"], field + ".gpu_count", 0),
        "gpu_memory_mib": _integer(
            payload["gpu_memory_mib"], field + ".gpu_memory_mib", 0
        ),
        "gpu_utilization_pct": _integer(
            payload["gpu_utilization_pct"],
            field + ".gpu_utilization_pct",
            0,
            100,
        ),
    }
    if not isinstance(payload["exclusive_gpu"], bool):
        raise AdapterError(field + ".exclusive_gpu must be boolean")
    normalized["exclusive_gpu"] = payload["exclusive_gpu"]
    return normalized


def _resource_hints(payload, field):
    if not isinstance(payload, dict):
        raise AdapterError("%s must be an object" % field)
    unknown = sorted(set(payload) - RESOURCE_FIELDS)
    if unknown:
        raise AdapterError(
            "%s contains unknown fields: %s" % (field, ",".join(unknown))
        )
    if not payload:
        raise AdapterError("%s must contain at least one hint" % field)
    normalized = {}
    for key in ("cpu_cores", "memory_mib"):
        if key in payload:
            normalized[key] = _integer(payload[key], field + "." + key, 1)
    if "gpu_count" in payload:
        normalized["gpu_count"] = _integer(
            payload["gpu_count"], field + ".gpu_count", 0
        )
    if "gpu_memory_mib" in payload:
        normalized["gpu_memory_mib"] = _integer(
            payload["gpu_memory_mib"], field + ".gpu_memory_mib", 0
        )
    if "gpu_utilization_pct" in payload:
        normalized["gpu_utilization_pct"] = _integer(
            payload["gpu_utilization_pct"],
            field + ".gpu_utilization_pct",
            0,
            100,
        )
    if "exclusive_gpu" in payload:
        if not isinstance(payload["exclusive_gpu"], bool):
            raise AdapterError(field + ".exclusive_gpu must be boolean")
        normalized["exclusive_gpu"] = payload["exclusive_gpu"]
    return normalized


def _validate_parameters(task, parameters):
    if task in MODEL_SIGNAL_TASKS:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.model_signal_ablation.protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    if task in SURVIVAL_RECOVERY_TASKS:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.survival.recovery_protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    if task in SURVIVAL_TASKS:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.survival.protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    if task in FEDFISHER_TIMING_TASKS:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.fedfisher_timing.protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    if task == EFFECTS_REPLAY_PILOT_TASK:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.one_shot_effects.diagnostics.protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    if task == ONE_SHOT_EFFECTS_PILOT_TASK:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.one_shot_effects.protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    if task in COMPARISON_TASKS:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from effect_modifier_experiment.causal_comparison.protocol import validate_parameters
        try:
            return validate_parameters(task, parameters)
        except (ValueError, TypeError) as error:
            raise AdapterError(str(error)) from error
    expected = frozenset(("variant", "shard_index", "shard_count"))
    _only_keys(parameters, expected, "parameters")
    variant = parameters["variant"]
    if variant not in VARIANTS:
        raise AdapterError("parameters.variant is not registered")
    shard_index = _integer(parameters["shard_index"], "parameters.shard_index", 0)
    shard_count = _integer(parameters["shard_count"], "parameters.shard_count", 1)
    allowed_counts = (1, 2, 4) if task == DSLASSO_TASK else (8, 16)
    if shard_count not in allowed_counts:
        raise AdapterError(
            "parameters.shard_count must be one of %s"
            % ",".join(str(value) for value in allowed_counts)
        )
    if shard_index >= shard_count:
        raise AdapterError("parameters.shard_index must be less than shard_count")
    return {
        "variant": variant,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }


def _absolute_log_path(value, field):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise AdapterError("%s must be an absolute path" % field)
    return value


def validate_running_job(payload, environment, manifest_path):
    """Validate one ServerScheduler running-job document and its allocation env."""
    _only_keys(payload, JOB_FIELDS, "running job")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise AdapterError("schema_version must be %d" % SCHEMA_VERSION)
    job_id = _identifier(payload["job_id"], "job_id")
    if payload["project"] != PROJECT_NAME:
        raise AdapterError("project must be FedFisher")
    task = _identifier(payload["task"], "task")
    if task not in ALLOWED_TASKS:
        raise AdapterError("task is not registered by the FedFisher adapter")
    if payload["state"] != "running":
        raise AdapterError("job state must be running")
    _timestamp(payload["submitted_at"], "submitted_at")
    _timestamp(payload["updated_at"], "updated_at")
    _integer(payload["priority"], "priority", -100, 100)
    execution_profile = _identifier(
        payload["execution_profile"], "execution_profile"
    )
    if execution_profile != TASK_PROFILES[task]["execution_profile"]:
        raise AdapterError("execution_profile is incompatible with the task")
    requested_profile = payload["requested_profile"]
    if requested_profile is not None:
        requested_profile = _identifier(requested_profile, "requested_profile")
        if requested_profile != execution_profile:
            raise AdapterError("requested_profile does not match execution_profile")
    estimated_runtime_seconds = _number(
        payload["estimated_runtime_seconds"],
        "estimated_runtime_seconds",
        0.0,
        strict_minimum=True,
    )
    parameters = _validate_parameters(task, payload["parameters"])
    requested = (
        None
        if payload["resources"] is None
        else _resource_hints(payload["resources"], "resources")
    )
    if payload["allocation"] is None:
        raise AdapterError("allocation is required")
    allocation = _resource(payload["allocation"], "allocation")
    expected_memory = TASK_PROFILES[task]["memory_mib"]
    if allocation["cpu_cores"] != 1:
        raise AdapterError("allocation.cpu_cores must be exactly 1")
    if allocation["memory_mib"] != expected_memory:
        raise AdapterError(
            "allocation.memory_mib must be exactly %d" % expected_memory
        )
    if (
        allocation["gpu_count"] != 0
        or allocation["gpu_memory_mib"] != 0
        or allocation["gpu_utilization_pct"] != 0
    ):
        raise AdapterError("allocation must describe a CPU-only allocation")
    if requested is not None:
        if requested.get("cpu_cores", 1) != 1:
            raise AdapterError("resources.cpu_cores must be exactly 1")
        if requested.get("memory_mib", expected_memory) != expected_memory:
            raise AdapterError(
                "resources.memory_mib must be exactly %d" % expected_memory
            )
        if (
            requested.get("gpu_count", 0) != 0
            or requested.get("gpu_memory_mib", 0) != 0
            or requested.get("gpu_utilization_pct", 0) != 0
        ):
            raise AdapterError("resources must describe a CPU-only request")
    if (
        requested is not None
        and "memory_mib" in requested
        and allocation["memory_mib"] < requested["memory_mib"]
    ):
        raise AdapterError("allocation.memory_mib may not be below the request")
    cpu_ids = payload["cpu_ids"]
    if not isinstance(cpu_ids, list):
        raise AdapterError("cpu_ids must be an array")
    cpu_ids = tuple(_integer(value, "cpu_ids", 0) for value in cpu_ids)
    if len(cpu_ids) != allocation["cpu_cores"] or len(set(cpu_ids)) != len(cpu_ids):
        raise AdapterError("cpu_ids must be unique and match allocation.cpu_cores")
    numa_node = payload["numa_node"]
    if numa_node is not None:
        numa_node = _integer(numa_node, "numa_node", 0)
    for field in ("gpu_indices", "gpu_uuids", "gpu_pci_bus_ids"):
        if not isinstance(payload[field], list):
            raise AdapterError("%s must be an array" % field)
        if payload[field]:
            raise AdapterError("CPU tasks require an empty %s array" % field)
    stdout_log = _absolute_log_path(payload["stdout_log"], "stdout_log")
    stderr_log = _absolute_log_path(payload["stderr_log"], "stderr_log")
    if payload["exit_code"] is not None or payload["failure_reason"] is not None:
        raise AdapterError("a running job may not already have completion fields")

    manifest_path = Path(manifest_path)
    expected_environment = {
        "SERVER_SCHEDULER_JOB_ID": job_id,
        "SERVER_SCHEDULER_PROJECT": PROJECT_NAME,
        "SERVER_SCHEDULER_JOB_MANIFEST": str(manifest_path.resolve(strict=True)),
        "SERVER_SCHEDULER_EXECUTION_PROFILE": execution_profile,
        "SERVER_SCHEDULER_CPU_CORES": str(allocation["cpu_cores"]),
        "SERVER_SCHEDULER_CPUSET": ",".join(str(value) for value in cpu_ids),
        "SERVER_SCHEDULER_MEMORY_MIB": str(allocation["memory_mib"]),
        "SERVER_SCHEDULER_GPU_COUNT": "0",
        "SERVER_SCHEDULER_GPU_MEMORY_MIB": "0",
        "SERVER_SCHEDULER_GPU_UTILIZATION_PCT": "0",
        "SERVER_SCHEDULER_GPU_EXCLUSIVITY": (
            "exclusive" if allocation["exclusive_gpu"] else "shared"
        ),
        "SERVER_SCHEDULER_GPU_INDICES": "",
        "SERVER_SCHEDULER_GPU_UUIDS": "",
        "SERVER_SCHEDULER_GPU_PCI_BUS_IDS": "",
        "SERVER_SCHEDULER_STDOUT_LOG": stdout_log,
        "SERVER_SCHEDULER_STDERR_LOG": stderr_log,
    }
    if numa_node is not None:
        expected_environment["SERVER_SCHEDULER_NUMA_NODE"] = str(numa_node)
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise AdapterError("%s is missing or inconsistent" % key)
    _identifier(environment.get("SERVER_SCHEDULER_LEASE_ID"), "SERVER_SCHEDULER_LEASE_ID")
    _environment_integer(environment, "SERVER_SCHEDULER_ATTEMPT", 1)
    environment_runtime = _environment_number(
        environment,
        "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS",
        0.0,
        strict_minimum=True,
    )
    if not math.isclose(
        environment_runtime,
        estimated_runtime_seconds,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise AdapterError(
            "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS disagrees with the manifest"
        )
    if numa_node is None and "SERVER_SCHEDULER_NUMA_NODE" in environment:
        raise AdapterError("SERVER_SCHEDULER_NUMA_NODE disagrees with the manifest")
    for key in (
        "SERVER_SCHEDULER_GPU_INDEX",
        "SERVER_SCHEDULER_GPU_UUID",
        "SERVER_SCHEDULER_GPU_PCI_BUS_ID",
    ):
        if key in environment:
            raise AdapterError("protocol v2 does not define %s" % key)
    if environment.get("CUDA_VISIBLE_DEVICES", ""):
        raise AdapterError("CPU task received nonempty CUDA_VISIBLE_DEVICES")
    if environment.get("NVIDIA_VISIBLE_DEVICES") != "void":
        raise AdapterError("CPU task requires NVIDIA_VISIBLE_DEVICES=void")
    allowed_scheduler_environment = set(expected_environment) | {
        "SERVER_SCHEDULER_LEASE_ID",
        "SERVER_SCHEDULER_ATTEMPT",
        "SERVER_SCHEDULER_ESTIMATED_RUNTIME_SECONDS",
    }
    unknown_scheduler_environment = sorted(
        key
        for key in environment
        if key.startswith("SERVER_SCHEDULER_")
        and key not in allowed_scheduler_environment
    )
    if unknown_scheduler_environment:
        raise AdapterError(
            "unexpected ServerScheduler environment variables: %s"
            % ",".join(unknown_scheduler_environment)
        )
    normalized = dict(payload)
    normalized["parameters"] = parameters
    normalized["resources"] = requested
    normalized["allocation"] = allocation
    normalized["requested_profile"] = requested_profile
    normalized["execution_profile"] = execution_profile
    normalized["estimated_runtime_seconds"] = estimated_runtime_seconds
    normalized["cpu_ids"] = cpu_ids
    normalized["numa_node"] = numa_node
    return normalized


def load_running_job(manifest_path, environment=None):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        raise AdapterError("job manifest path must be absolute")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AdapterError("job manifest must be a regular non-symlink file")
    if manifest_path.resolve(strict=True) != manifest_path.absolute():
        raise AdapterError("job manifest path may not resolve through symlinks")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise AdapterError("job manifest exceeds 1 MiB")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError("non-finite JSON value: %s" % value)
                ),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AdapterError("cannot read strict JSON job manifest: %s" % error) from error
    return validate_running_job(payload, os.environ if environment is None else environment, manifest_path)


def _safe_runtime(job_id, cpu_cores):
    source_root = Path(__file__).resolve().parents[1]
    if source_root != PROJECT_ROOT.resolve(strict=True):
        raise AdapterError("adapter must execute from the registered FedFisher code root")
    runtime = SCRATCH_ROOT / "effect_modifier_experiment" / "server_scheduler" / job_id
    root = SCRATCH_ROOT.resolve(strict=True)
    if runtime.resolve(strict=False) != runtime or root not in runtime.parents:
        raise AdapterError("unsafe ServerScheduler runtime path")
    current = runtime
    while current != root:
        if current.exists() and current.is_symlink():
            raise AdapterError("runtime path may not contain symlinks")
        current = current.parent
    runtime.mkdir(parents=True, exist_ok=True)
    for path in (runtime,):
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise AdapterError("runtime path may not resolve through symlinks")
    for name in ("tmp", "python_cache", "xdg_cache", "torch_cache"):
        child = runtime / name
        child.mkdir(exist_ok=True)
        if child.is_symlink() or child.resolve(strict=True) != child:
            raise AdapterError("runtime child may not resolve through symlinks")
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(cpu_cores),
            "MKL_NUM_THREADS": str(cpu_cores),
            "OPENBLAS_NUM_THREADS": str(cpu_cores),
            "NUMEXPR_NUM_THREADS": str(cpu_cores),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TMPDIR": str(runtime / "tmp"),
            "PYTHONPYCACHEPREFIX": str(runtime / "python_cache"),
            "XDG_CACHE_HOME": str(runtime / "xdg_cache"),
            "TORCH_HOME": str(runtime / "torch_cache"),
        }
    )


def _dslasso_runner_args(runner, run_id, variant, output_dir):
    values = DSLASSO_ENVIRONMENT
    argv = [
        "--run-id", run_id,
        "--variant", variant,
        "--output-dir", str(output_dir),
        "--seed", "0",
        "--split", "iid",
        "--alpha", values["SYNTHETIC_ALPHA"],
        "--num-sites", values["EFFECT_MODIFIER_NUM_SITES"],
        "--num-test", values["SYNTHETIC_NUM_TEST"],
        "--covariate-dim", values["EFFECT_MODIFIER_COVARIATE_DIM"],
        "--signal-dim", values["EFFECT_MODIFIER_SIGNAL_DIM"],
        "--signal-strength", values["EFFECT_MODIFIER_SIGNAL_STRENGTH"],
        "--main-effect-strength", values["EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH"],
        "--confounding-strength", values["EFFECT_MODIFIER_CONFOUNDING_STRENGTH"],
        "--outcome-intercept", values["EFFECT_MODIFIER_INTERCEPT"],
        "--treatment-prob", values["EFFECT_MODIFIER_TREATMENT_PROB"],
        "--propensity-clip", values["EFFECT_MODIFIER_PROPENSITY_CLIP"],
        "--site-min-samples", values["EFFECT_MODIFIER_SITE_MIN_SAMPLES"],
        "--site-max-samples", values["EFFECT_MODIFIER_SITE_MAX_SAMPLES"],
        "--site-gaussian-mean-span", values["EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN"],
        "--site-gaussian-std-low", values["EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW"],
        "--site-gaussian-std-high", values["EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH"],
        "--intervention-modes",
        *values["PREDICTION_INTERVENTION_MODES"].split(","),
        "--intervention-repeats", values["PREDICTION_INTERVENTION_REPEATS"],
        "--intervention-batch-size", values["PREDICTION_INTERVENTION_BATCH_SIZE"],
        "--cv-folds", values["CV_FOLDS"],
        "--cv-seed-offset", values["CV_SEED_OFFSET"],
        "--lambda-count", values["LAMBDA_COUNT"],
        "--lambda-min-ratio", values["LAMBDA_MIN_RATIO"],
        "--standardization-scale-floor", values["STANDARDIZATION_SCALE_FLOOR"],
        "--optimizer-max-iter", values["OPTIMIZER_MAX_ITER"],
        "--optimizer-tolerance", values["OPTIMIZER_TOLERANCE"],
        "--optimizer-initial-step", values["OPTIMIZER_INITIAL_STEP"],
        "--optimizer-backtrack-factor", values["OPTIMIZER_BACKTRACK_FACTOR"],
        "--optimizer-min-step", values["OPTIMIZER_MIN_STEP"],
        "--threads-per-worker", "1",
        "--cpu-workers", os.environ["CPU_WORKERS"],
    ]
    return runner.build_parser().parse_args(argv)


def _run_dslasso(job):
    from effect_modifier_experiment import manage_dslassocov_cpu_run as manager
    from effect_modifier_experiment import run_dslassocov_effect_modifier as runner

    parameters = job["parameters"]
    variant = parameters["variant"]
    shard_index = parameters["shard_index"]
    shard_count = parameters["shard_count"]
    run_id = DSLASSO_RUN_IDS[variant]
    os.environ.update(DSLASSO_ENVIRONMENT)
    os.environ.update(
        {
            "RUN_ID": run_id,
            "EFFECT_MODIFIER_VARIANT": variant,
            "CPU_WORKERS": str(shard_count),
        }
    )
    paths, unused_manifest = manager.initialize_run(
        run_id, variant, create_legacy_scheduler_dirs=False
    )
    del unused_manifest
    runner_args = _dslasso_runner_args(runner, run_id, variant, paths["output_dir"])
    for seed in range(shard_index, 300, shard_count):
        manager.validate_current_run(run_id, variant)
        status_code = manager.seed_status(run_id, variant, seed)
        if status_code == 0:
            continue
        if status_code != 3:
            raise AdapterError("dsLassoCov seed has partial or invalid output")
        for split in ("iid", "noniid"):
            runner_args.seed = seed
            runner_args.split = split
            runner.run(runner_args)
        manager.complete_seed(run_id, variant, seed)
    return 0


def _run_one_shot(job):
    from effect_modifier_experiment import manage_one_shot_lasso_cpu_run as manager
    from effect_modifier_experiment import run_one_shot_lasso_effect_modifier as runner

    parameters = job["parameters"]
    variant = parameters["variant"]
    shard_index = parameters["shard_index"]
    shard_count = parameters["shard_count"]
    for family in manager.FAMILIES:
        manager.initialize_formal(
            manager.formal_runner_args(
                family, variant, run_suffix=ONE_SHOT_RUN_SUFFIX
            )
        )
    for seed in manager.shard_seed_sequence(shard_index, shard_count):
        for family in manager.FAMILIES:
            for split in ("iid", "noniid"):
                for model in manager.MODELS[family]:
                    args = manager.formal_runner_args(
                        family,
                        variant,
                        seed,
                        split,
                        model,
                        run_suffix=ONE_SHOT_RUN_SUFFIX,
                    )
                    status_code = manager.cell_status(args)
                    if status_code == 0:
                        continue
                    if status_code != manager.MISSING_EXIT:
                        raise AdapterError("one-shot cell has partial or invalid output")
                    runner.run(args)
                    if manager.cell_status(args) != 0:
                        raise AdapterError("one-shot completion marker did not validate")
    return 0


def execute_task(job):
    _safe_runtime(job["job_id"], job["allocation"]["cpu_cores"])
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if job["task"] == DSLASSO_TASK:
        return _run_dslasso(job)
    if job["task"] == ONE_SHOT_TASK:
        return _run_one_shot(job)
    if job["task"] in COMPARISON_TASKS:
        from effect_modifier_experiment.causal_comparison.runner import run_shard
        return run_shard(job)
    if job["task"] == ONE_SHOT_EFFECTS_PILOT_TASK:
        from effect_modifier_experiment.one_shot_effects.runner import run_job
        return run_job(job)
    if job["task"] == EFFECTS_REPLAY_PILOT_TASK:
        from effect_modifier_experiment.one_shot_effects.diagnostics.runner import run_job
        return run_job(job)
    if job["task"] in FEDFISHER_TIMING_TASKS:
        from effect_modifier_experiment.fedfisher_timing.runner import run_shard
        return run_shard(job)
    if job["task"] in MODEL_SIGNAL_TASKS:
        from effect_modifier_experiment.model_signal_ablation.runner import run_shard
        return run_shard(job)
    if job["task"] in SURVIVAL_TASKS:
        from effect_modifier_experiment.survival.runner import run_job
        return run_job(job)
    if job["task"] in SURVIVAL_RECOVERY_TASKS:
        from effect_modifier_experiment.survival.recovery import run_job
        return run_job(job)
    raise AdapterError("unreachable task dispatch")


def run_from_manifest(manifest_path, environment=None, executor=execute_task):
    job = load_running_job(manifest_path, environment)
    return executor(job)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-manifest", required=True, type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run_from_manifest(args.job_manifest)
    except AdapterError as error:
        print("fedfisher_server_scheduler_adapter_error=%s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
