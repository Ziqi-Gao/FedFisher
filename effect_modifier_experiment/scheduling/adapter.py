"""Foreground ServerScheduler adapter for reviewed FedFisher CPU shards."""
import argparse
from pathlib import Path
import sys

from .catalog import (
    PROJECT_ROOT, SCRATCH_ROOT, PROJECT_NAME,
    SCHEMA_VERSION, CPU_PROFILE, IDENTIFIER,
    VARIANTS, DSLASSO_TASK, ONE_SHOT_TASK,
    COMPARISON_PILOT_TASK, COMPARISON_TASK, ONE_SHOT_EFFECTS_PILOT_TASK,
    EFFECTS_REPLAY_PILOT_TASK, FEDFISHER_TIMING_PILOT_TASK, FEDFISHER_TIMING_TASK,
    MODEL_SIGNAL_PILOT_TASK, MODEL_SIGNAL_TASK, MODEL_SIGNAL_RECOVERY_PILOT_TASK,
    MODEL_SIGNAL_RECOVERY_TASK, SURVIVAL_PILOT_TASK, SURVIVAL_TASK,
    SURVIVAL_TASKS, SURVIVAL_DIAGNOSTIC_TASK, SURVIVAL_RECOVERY_TASK,
    SURVIVAL_REPAIR_PILOT_TASK, SURVIVAL_REPAIR_TASK, SURVIVAL_INTEGRATED_PILOT_TASK,
    SURVIVAL_INTEGRATED_TASK, SURVIVAL_RECOVERY_TASKS, COMPARISON_TASKS,
    FEDFISHER_TIMING_TASKS, MODEL_SIGNAL_TASKS, MODEL_SIGNAL_RECOVERY_TASKS,
    ALLOWED_TASKS, JOB_FIELDS, RESOURCE_FIELDS,
    TASK_PROFILES,
    TASK_CONTRACTS,
)
from .errors import AdapterError
from .legacy import (
    DSLASSO_ENVIRONMENT, DSLASSO_RUN_IDS, ONE_SHOT_RUN_SUFFIX,
    _dslasso_runner_args, _run_dslasso, _run_one_shot,
)
from .runtime import configure_runtime as _safe_runtime
from .dispatch import execute_task as dispatch_task
from .validation import load_running_job, validate_running_job


def execute_task(job):
    return dispatch_task(job, runtime=_safe_runtime)



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
