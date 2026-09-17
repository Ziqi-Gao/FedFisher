"""Immutable outbox publication; experiment callers own scientific recipes.

This module writes project request bundles only. It has no scheduler client,
resource selection, model imports, or authority to submit a request.
"""
import hashlib
import json
import os
from pathlib import Path
import re

from .artifacts import canonical
from .provenance import SCRATCH_ROOT


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or SCRATCH_ROOT not in path.parents:
        raise ValueError('request publication must stay below the FedFisher scratch root')
    if any(p.is_symlink() for p in (path, *path.parents)) or path.resolve(strict=False) != path:
        raise ValueError('publication path must not contain symlinks')
    return path


def pending_path(path):
    return path.with_name('.' + path.name + '.pending')


def preflight_new(path):
    path = safe_path(path)
    for target in (path, safe_path(pending_path(path))):
        if target.exists():
            raise FileExistsError('preserving existing or partial publication: %s' % target)
    return path


def atomic_create(path, data):
    path = preflight_new(path)
    pending = pending_path(path)
    fd = os.open(str(pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    safe_path(path)
    safe_path(pending)
    os.link(pending, path)
    pending.unlink()


def clean_provenance(value):
    """Validate a captured provenance record without choosing its source root."""
    if value.get('git_dirty') != '0':
        raise ValueError('request publication requires finalized clean source; dirty tree refused')
    if re.fullmatch('[0-9a-f]{40}', value.get('git_commit', '')) is None:
        raise ValueError('invalid source commit')
    for key in ('git_status_sha256', 'source_tree_sha256', 'provenance_sha256'):
        if re.fullmatch('[0-9a-f]{64}', value.get(key, '')) is None:
            raise ValueError('invalid source provenance: ' + key)
    return dict(value)


def review_sources(paths, project_root):
    result = {}
    for label, path in paths.items():
        path = Path(path)
        if (project_root not in path.parents or path.resolve(strict=True) != path
                or any(p.is_symlink() for p in (path, *path.parents))):
            raise ValueError('review source must be a regular canonical project file')
        result[label] = dict(path=str(path), sha256=sha256(path.read_bytes()))
    return result


def publish_bundle(specs, *, bundle, inventory_path, metadata, provenance, sources,
                   acceptance, check_provenance, check_sources, check_acceptance,
                   create=atomic_create):
    """Verify actual request bytes and stable bindings, then publish inventory.

Callers validate scientific scope and pilot prerequisites before entering this
function. Recheck callbacks retain each experiment's own acceptance policy.
Partial writes are preserved and a retry cannot overwrite them.
"""
    bundle, inventory_path = safe_path(bundle), safe_path(inventory_path)
    if bundle.exists():
        raise FileExistsError('preserving existing or partial cohort directory: %s' % bundle)
    targets = []
    for item in specs:
        name = item['filename']
        if not isinstance(name, str) or Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('request filename must be a single ordinary basename')
        targets.append(bundle / name)
    if not targets or len(set(targets)) != len(targets) or inventory_path in targets:
        raise ValueError('empty bundle or duplicate publication target')
    for path in (*targets, inventory_path):
        preflight_new(path)
    bundle.mkdir(parents=True, exist_ok=False)
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path(bundle)
    safe_path(inventory_path.parent)
    records = []
    for item, path in zip(specs, targets):
        create(path, encoded(item['request']))
        record = dict(path=str(path), sha256=item['request_sha256'],
                      job_id=item['request']['job_id'], task=item['request']['task'],
                      parameters=item['request']['parameters'], run_id=item['run_id'],
                      configuration=item['configuration'],
                      configuration_sha256=item['configuration_sha256'], seeds=item['seed_sequence'])
        if 'splits' in item:
            record['splits'] = item['splits']
        records.append(record)
    for item, path in zip(specs, targets):
        actual = safe_path(path).read_bytes()
        if actual != encoded(item['request']) or sha256(actual) != item['request_sha256']:
            raise ValueError('request readback mismatch; partial cohort preserved')
    if check_provenance() != provenance or check_sources() != sources:
        raise ValueError('source changed during publication; partial cohort preserved')
    if check_acceptance() != acceptance:
        raise ValueError('pilot proof changed; pilot acceptance changed during publication; partial cohort preserved')
    inventory = dict(metadata, request_count=len(records), bundle_ready=True, submitted=False,
                     requires_operator_registration=True,
                     publication_mode='nested_operator_review_bundle_not_automatic_intake',
                     registration_state='project proposal only; no central registration or admission implied',
                     provenance=provenance, project_head=provenance['git_commit'], worktree_status='clean',
                     review_sources=sources, requests=records, inventory_path=str(inventory_path))
    if acceptance is not None:
        inventory['pilot_acceptance'] = acceptance
    create(inventory_path, encoded(inventory))
    if inventory_path.read_bytes() != encoded(inventory):
        raise ValueError('inventory readback mismatch')
    return inventory
