#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"

fail() {
    echo "$*" >&2
    exit 2
}

canonical_exact() {
    local label="$1"
    local candidate="$2"
    local expected="$3"
    local resolved_candidate
    local resolved_expected

    resolved_candidate="$(realpath -m -- "${candidate}")"
    resolved_expected="$(realpath -e -- "${expected}")"
    [[ "${resolved_candidate}" == "${resolved_expected}" ]] \
        || fail "${label} must resolve to ${expected}: ${resolved_candidate}"
    printf '%s\n' "${resolved_candidate}"
}

canonical_bound_path() {
    local label="$1"
    local candidate="$2"
    local allowed_root="$3"
    local relative_path="$4"
    local resolved_candidate
    local resolved_root
    local expected

    resolved_candidate="$(realpath -m -- "${candidate}")"
    resolved_root="$(realpath -e -- "${allowed_root}")"
    expected="${resolved_root}/${relative_path}"
    [[ "${resolved_candidate}" == "${expected}" ]] \
        || fail "${label} must resolve to ${expected}: ${resolved_candidate}"
    printf '%s\n' "${resolved_candidate}"
}

command -v realpath >/dev/null 2>&1 \
    || fail "Required command is unavailable: realpath"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(canonical_exact \
    "REPO_ROOT" \
    "$(cd -- "${SCRIPT_DIR}/.." && pwd -P)" \
    "${FEDFISHER_PROJECT_ROOT}")"
SCRATCH_ROOT="$(canonical_exact \
    "SCRATCH_ROOT" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "${FEDFISHER_SCRATCH_ROOT}")"
ENV_DIR="$(canonical_bound_path \
    "ENV_DIR" \
    "${SCRATCH_ROOT}/envs/fedfisher-py39" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "envs/fedfisher-py39")"
PIP_CACHE_DIR="$(canonical_bound_path \
    "PIP_CACHE_DIR" \
    "${SCRATCH_ROOT}/pip-cache" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "pip-cache")"
TMPDIR="$(canonical_bound_path \
    "TMPDIR" \
    "${SCRATCH_ROOT}/tmp" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "tmp")"
SETUP_RUNTIME="$(canonical_bound_path \
    "setup runtime" \
    "${SCRATCH_ROOT}/setup-runtime" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "setup-runtime")"

mkdir -p "${ENV_DIR%/*}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${SETUP_RUNTIME}"
ENV_DIR="$(canonical_bound_path "ENV_DIR" "${ENV_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "envs/fedfisher-py39")"
PIP_CACHE_DIR="$(canonical_bound_path "PIP_CACHE_DIR" "${PIP_CACHE_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "pip-cache")"
TMPDIR="$(canonical_bound_path "TMPDIR" "${TMPDIR}" "${FEDFISHER_SCRATCH_ROOT}" "tmp")"
SETUP_RUNTIME="$(canonical_bound_path "setup runtime" "${SETUP_RUNTIME}" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime")"

export PIP_CACHE_DIR TMPDIR
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
XDG_CACHE_HOME="$(canonical_bound_path "XDG_CACHE_HOME" "${SETUP_RUNTIME}/xdg_cache" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/xdg_cache")"
PYTHONPYCACHEPREFIX="$(canonical_bound_path "PYTHONPYCACHEPREFIX" "${SETUP_RUNTIME}/python_cache" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/python_cache")"
CUDA_CACHE_PATH="$(canonical_bound_path "CUDA_CACHE_PATH" "${SETUP_RUNTIME}/cuda_cache" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/cuda_cache")"
TRITON_CACHE_DIR="$(canonical_bound_path "TRITON_CACHE_DIR" "${SETUP_RUNTIME}/triton_cache" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/triton_cache")"
TORCH_HOME="$(canonical_bound_path "TORCH_HOME" "${SETUP_RUNTIME}/torch_cache" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/torch_cache")"
export XDG_CACHE_HOME PYTHONPYCACHEPREFIX CUDA_CACHE_PATH TRITON_CACHE_DIR TORCH_HOME
mkdir -p \
    "${XDG_CACHE_HOME}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${CUDA_CACHE_PATH}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_HOME}"
canonical_bound_path "XDG_CACHE_HOME" "${XDG_CACHE_HOME}" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/xdg_cache" >/dev/null
canonical_bound_path "PYTHONPYCACHEPREFIX" "${PYTHONPYCACHEPREFIX}" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/python_cache" >/dev/null
canonical_bound_path "CUDA_CACHE_PATH" "${CUDA_CACHE_PATH}" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/cuda_cache" >/dev/null
canonical_bound_path "TRITON_CACHE_DIR" "${TRITON_CACHE_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/triton_cache" >/dev/null
canonical_bound_path "TORCH_HOME" "${TORCH_HOME}" "${FEDFISHER_SCRATCH_ROOT}" "setup-runtime/torch_cache" >/dev/null

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    python3 -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install --upgrade -r "${REPO_ROOT}/requirements.txt"
"${ENV_DIR}/bin/python" - <<'PY_CHECK'
import torch
import nngeometry
import numpy

assert torch.cuda.is_available(), "CUDA is not available"
assert "sm_120" in torch.cuda.get_arch_list(), "PyTorch wheel does not include Blackwell sm_120 kernels"
x = torch.ones(1, device="cuda")
assert (x + x).cpu().item() == 2.0
print("environment_ok")
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("nngeometry", nngeometry.__file__)
PY_CHECK

printf 'Environment ready: %s\n' "${ENV_DIR}"
