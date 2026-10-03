#!/usr/bin/env python3
"""Run bounded development fixtures with one thread and project-scratch caches."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path('/home/del6500/projects/FedFisher')
SCRATCH_ROOT = Path('/scr/del6500/FedFisher')


def source_directory():
    """Allow canonical fixtures and isolated Git review fixtures only."""
    source = Path(__file__).resolve().parents[1]
    if source == PROJECT_ROOT:
        return source
    review = PROJECT_ROOT / '.git/codex-review'
    metadata = source / '.git'
    if not source.is_relative_to(review) or not metadata.is_file() or metadata.is_symlink():
        raise ValueError('fixtures require the canonical tree or an isolated review worktree')
    value = metadata.read_text().strip()
    if not value.startswith('gitdir: '):
        raise ValueError('review worktree has invalid Git metadata')
    gitdir = Path(value[len('gitdir: '):]).resolve(strict=True)
    if not gitdir.is_relative_to(PROJECT_ROOT / '.git/worktrees'):
        raise ValueError('review worktree metadata is outside the canonical repository')
    return source


def scratch_directory():
    root = SCRATCH_ROOT / 'tests'
    if any(p.is_symlink() for p in (root, *root.parents)) or root.resolve() != root:
        raise ValueError('test scratch must be an ordinary project directory')
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root:
        raise ValueError('test scratch changed during creation')
    return Path(tempfile.mkdtemp(prefix='suite-', dir=root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='test_*.py', help='unittest filename pattern')
    args = parser.parse_args(argv)
    try:
        source = source_directory()
    except ValueError as error:
        parser.error(str(error))
    scratch = scratch_directory()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(source), TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch),
        XDG_CACHE_HOME=str(scratch), MPLCONFIGDIR=str(scratch / 'matplotlib'),
        TORCH_HOME=str(scratch / 'torch'), HF_HOME=str(scratch / 'huggingface'),
        TRITON_CACHE_DIR=str(scratch / 'triton'), NUMBA_CACHE_DIR=str(scratch / 'numba'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        environment[key] = '1'
    print('Development fixtures; scratch: ' + str(scratch), flush=True)
    return subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
        '-s', 'tests', '-t', '.', '-p', args.pattern, '-v'], cwd=source, env=environment).returncode


if __name__ == '__main__':
    raise SystemExit(main())
