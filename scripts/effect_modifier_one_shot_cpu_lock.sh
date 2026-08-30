#!/usr/bin/env bash

# Source-only lock helper for the formal one-shot LASSO CPU scheduler.
# This namespace is separate from both the GPU scheduler and dsLassoCov locks.

readonly ONE_SHOT_CPU_LOCK_ROOT="/scr/del6500/FedFisher/effect_modifier_experiment/one_shot_cpu_locks"
readonly ONE_SHOT_LOCK_BUSY_EXIT=73

ONE_SHOT_CPU_LOCK_FD=""
ONE_SHOT_CPU_LOCK_PATH=""

_one_shot_validate_variant() {
    case "$1" in
        control|site_gaussian_shift|local_permutation|site_gaussian_local_permutation) ;;
        *) echo "Unsafe one-shot lock variant: $1" >&2; return 2 ;;
    esac
}

one_shot_prepare_lock_root() {
    local parent
    parent="$(dirname -- "${ONE_SHOT_CPU_LOCK_ROOT}")"
    [[ -d "${parent}" && ! -L "${parent}" ]] || {
        echo "One-shot lock parent is missing or unsafe: ${parent}" >&2
        return 2
    }
    [[ "$(realpath -e -- "${parent}")" == "${parent}" ]] || {
        echo "One-shot lock parent may not resolve through a symlink" >&2
        return 2
    }
    if [[ ! -e "${ONE_SHOT_CPU_LOCK_ROOT}" ]]; then
        mkdir -- "${ONE_SHOT_CPU_LOCK_ROOT}"
        chmod 700 "${ONE_SHOT_CPU_LOCK_ROOT}"
    fi
    [[ -d "${ONE_SHOT_CPU_LOCK_ROOT}" && ! -L "${ONE_SHOT_CPU_LOCK_ROOT}" ]] || {
        echo "One-shot lock root is unsafe: ${ONE_SHOT_CPU_LOCK_ROOT}" >&2
        return 2
    }
    [[ "$(realpath -e -- "${ONE_SHOT_CPU_LOCK_ROOT}")" == "${ONE_SHOT_CPU_LOCK_ROOT}" ]] || {
        echo "One-shot lock root may not resolve through a symlink" >&2
        return 2
    }
}

one_shot_acquire_variant_lock() {
    local variant="$1"
    local owner="$2"
    local lock_path
    command -v flock >/dev/null 2>&1 || {
        echo "Required CPU lock command is unavailable: flock" >&2
        return 2
    }
    _one_shot_validate_variant "${variant}" || return $?
    one_shot_prepare_lock_root || return $?
    lock_path="${ONE_SHOT_CPU_LOCK_ROOT}/formal_stationarity_polish_${variant}.lock"
    [[ ! -L "${lock_path}" ]] || {
        echo "One-shot lock file may not be a symlink: ${lock_path}" >&2
        return 2
    }
    exec {ONE_SHOT_CPU_LOCK_FD}>>"${lock_path}"
    if ! flock --exclusive --nonblock "${ONE_SHOT_CPU_LOCK_FD}"; then
        echo "one_shot_cpu_lock_status=busy path=${lock_path}" >&2
        sed -n '1,40p' "${lock_path}" >&2 || true
        exec {ONE_SHOT_CPU_LOCK_FD}>&-
        ONE_SHOT_CPU_LOCK_FD=""
        return "${ONE_SHOT_LOCK_BUSY_EXIT}"
    fi
    ONE_SHOT_CPU_LOCK_PATH="${lock_path}"
    printf 'event=acquired time=%s pid=%s owner=%s\n' \
        "$(date -Is)" "$$" "${owner}" >&"${ONE_SHOT_CPU_LOCK_FD}"
    echo "one_shot_cpu_lock_status=acquired path=${lock_path} pid=$$ owner=${owner}"
}

one_shot_release_cpu_lock() {
    local owner="${1:-unknown}"
    if [[ -n "${ONE_SHOT_CPU_LOCK_FD}" ]]; then
        printf 'event=released time=%s pid=%s owner=%s\n' \
            "$(date -Is)" "$$" "${owner}" >&"${ONE_SHOT_CPU_LOCK_FD}" || true
        flock --unlock "${ONE_SHOT_CPU_LOCK_FD}" || true
        exec {ONE_SHOT_CPU_LOCK_FD}>&-
        echo "one_shot_cpu_lock_status=released path=${ONE_SHOT_CPU_LOCK_PATH} pid=$$ owner=${owner}"
    fi
    ONE_SHOT_CPU_LOCK_FD=""
    ONE_SHOT_CPU_LOCK_PATH=""
}
