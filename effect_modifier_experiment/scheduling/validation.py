"""Pure protocol-v2 manifest/environment checks; no scientific runner imports."""
from datetime import datetime
from importlib import import_module
import json
import math
import os
from pathlib import Path
import re

from .catalog import (
    ALLOWED_TASKS, DSLASSO_TASK, IDENTIFIER, JOB_FIELDS, PROJECT_NAME,
    RESOURCE_FIELDS, SCHEMA_VERSION, TASK_CONTRACTS, TASK_PROFILES, VARIANTS,
)
from .errors import AdapterError


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
    contract = TASK_CONTRACTS.get(task)
    if contract is None:
        raise AdapterError('task is not registered by the FedFisher adapter')
    if contract.protocol_module:
        validate_parameters = import_module(contract.protocol_module).validate_parameters
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
