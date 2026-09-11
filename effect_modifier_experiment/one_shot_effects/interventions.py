"""Data-independent permutation plans shared by both local summary methods."""
import hashlib

import numpy as np


def make_permutations(n, d, seed, repeats=5):
    if any(type(value) is not int for value in (n, d, seed, repeats)):
        raise ValueError('permutation dimensions and seed must be integers')
    if min(n, d, repeats) < 1 or seed < 0:
        raise ValueError('invalid permutation dimensions or seed')
    rng = np.random.default_rng(seed)
    result = np.empty((d, repeats, n), dtype=np.int64)
    for j in range(d):
        for repeat in range(repeats):
            result[j, repeat] = rng.permutation(n)
    result.setflags(write=False)
    return result


def validate_permutations(x, plan):
    x, plan = np.asarray(x), np.asarray(plan)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('finite raw covariates required')
    if plan.ndim != 3 or plan.shape[0] != x.shape[1] or plan.shape[2] != len(x):
        raise ValueError('permutation dimensions disagree with covariates')
    if plan.dtype.kind not in 'iu' or plan.shape[1] < 1:
        raise ValueError('integer permutation plans required')
    expected = np.arange(len(x))
    if not np.all(np.sort(plan, axis=2) == expected):
        raise ValueError('every intervention must be a complete permutation')


def perturbed_covariates(x, plan):
    validate_permutations(x, plan)
    for j in range(x.shape[1]):
        for repeat in range(plan.shape[1]):
            changed = np.asarray(x).copy()
            changed[:, j] = x[plan[j, repeat], j]
            yield j, repeat, changed


def permutation_hash(plan):
    values = np.asarray(plan, dtype='<i8', order='C')
    return hashlib.sha256(str(values.shape).encode() + values.tobytes()).hexdigest()
