"""Create audited protocol-v2 FedFisher pilot or post-pilot request sets."""

import argparse
from effect_modifier_experiment.scheduling.publication import require_canonical_source

import hashlib

import json

from pathlib import Path

import subprocess

OUTBOX_ROOT = Path("/scr/del6500/FedFisher/scheduler/outbox")

from effect_modifier_experiment.requests import (
    PROJECT_ROOT,
    PROJECT,
    EPOCH,
    VARIANTS,
    TASKS,
    REVIEW_SOURCES,
    _job_id,
    build_requests,
    _encoded,
    review_sources,
)


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
    require_canonical_source(__file__, PROJECT_ROOT)
    head = _git_text("rev-parse", "HEAD")
    status = _git_text("status", "--porcelain=v1", "--untracked-files=all")
    return head, status


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
