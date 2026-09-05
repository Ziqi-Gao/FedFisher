#!/usr/bin/env python3
"""Create audited protocol-v2 FedFisher pilot or post-pilot request sets."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
OUTBOX_ROOT = Path("/scr/del6500/FedFisher/scheduler/outbox")
PROJECT = "FedFisher"
EPOCH = "v2r2"
VARIANTS = (
    "control",
    "site_gaussian_shift",
    "local_permutation",
    "site_gaussian_local_permutation",
)
TASKS = {
    "dslasso": {
        "task": "effect-modifier-dslassocov-shard",
        "shard_count": 4,
        "pilot_shards": {variant: 0 for variant in VARIANTS},
    },
    "one-shot": {
        "task": "effect-modifier-one-shot-lasso-shard",
        "shard_count": 16,
        "pilot_shards": {
            "control": 6,
            "site_gaussian_shift": 13,
            "local_permutation": 6,
            "site_gaussian_local_permutation": 13,
        },
    },
}


def _git_text(*args):
    completed = subprocess.run(
        ("git",) + args,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def git_snapshot():
    head = _git_text("rev-parse", "HEAD")
    status = _git_text("status", "--porcelain=v1", "--untracked-files=all")
    return head, status


def _job_id(task_label, variant, phase, generation, shard_index, shard_count):
    variant_label = variant.replace("_", "-")
    return (
        "fedfisher-%s-%s-%s-%s-%s-p%d-shard-%02d-of-%02d"
        % (
            task_label,
            variant_label,
            EPOCH,
            phase,
            "cohort",
            generation,
            shard_index,
            shard_count,
        )
    )


def build_requests(phase, generation=1):
    if phase not in {"pilot", "remaining"}:
        raise ValueError("phase must be pilot or remaining")
    if isinstance(generation, bool) or int(generation) < 1:
        raise ValueError("generation must be a positive integer")
    generation = int(generation)
    requests = []
    for task_label, contract in TASKS.items():
        shard_count = contract["shard_count"]
        for variant in VARIANTS:
            pilot_index = contract["pilot_shards"][variant]
            if phase == "pilot":
                shard_indices = (pilot_index,)
            else:
                shard_indices = tuple(
                    index for index in range(shard_count) if index != pilot_index
                )
            for shard_index in shard_indices:
                job_id = _job_id(
                    task_label,
                    variant,
                    phase,
                    generation,
                    shard_index,
                    shard_count,
                )
                payload = {
                    "schema_version": 2,
                    "job_id": job_id,
                    "project": PROJECT,
                    "task": contract["task"],
                    "parameters": {
                        "variant": variant,
                        "shard_index": shard_index,
                        "shard_count": shard_count,
                    },
                    "priority": 0,
                }
                requests.append((job_id + ".job-request.json", payload))
    return tuple(requests)


def _encoded(payload):
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def write_requests(phase, generation, confirm_pilots_validated=False):
    if phase == "remaining" and not confirm_pilots_validated:
        raise ValueError(
            "remaining requests require --confirm-pilots-validated after central "
            "completion and project scientific validation"
        )
    head, status = git_snapshot()
    if status:
        raise ValueError("refusing request generation from a dirty worktree")
    requests = build_requests(phase, generation)
    OUTBOX_ROOT.mkdir(parents=True, exist_ok=True)
    targets = tuple(OUTBOX_ROOT / name for name, _ in requests)
    existing = tuple(path for path in targets if path.exists() or path.is_symlink())
    if existing:
        raise FileExistsError(
            "refusing to overwrite request files: %s"
            % ", ".join(str(path) for path in existing)
        )
    records = []
    for (name, payload), path in zip(requests, targets):
        body = _encoded(payload)
        with path.open("xb") as handle:
            handle.write(body)
        records.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(body).hexdigest(),
                "job_id": payload["job_id"],
            }
        )
    return {
        "project_head": head,
        "worktree_status": "clean",
        "phase": phase,
        "generation": generation,
        "request_count": len(records),
        "submitted": False,
        "requests": records,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument(
        "--confirm-pilots-validated",
        action="store_true",
        help="required for remaining requests after central and scientific pilot checks",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        summary = write_requests(
            args.phase,
            args.generation,
            confirm_pilots_validated=args.confirm_pilots_validated,
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print("request_generation_error=%s" % error)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
