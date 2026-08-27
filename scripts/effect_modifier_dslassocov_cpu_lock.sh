#!/usr/bin/env bash

# Source-only lock helpers for the CPU dsLassoCov-style scheduler.
# This namespace is intentionally separate from effect_modifier_experiment/locks,
# which belongs exclusively to the existing GPU scheduler.

readonly DSLASSOCOV_CPU_LOCK_ROOT="/scr/del6500/FedFisher/effect_modifier_experiment/cpu_locks"
readonly DSLASSOCOV_LOCK_BUSY_EXIT=73

DSLASSOCOV_CPU_LOCK_FD=""
DSLASSOCOV_CPU_LOCK_PATH=""

_dslassocov_validate_lock_run_id() {
    local run_id="$1"
    [[ "${run_id}" =~ ^dslassocov_(control|site_gaussian_shift|local_permutation|site_gaussian_local_permutation)_[A-Za-z0-9._-]+$ ]] || {
        echo "Unsafe dsLassoCov RUN_ID for CPU lock: ${run_id}" >&2
        return 2
    }
}

_dslassocov_validate_lock_dir() {
    local run_id="$1"
    local lock_dir="${DSLASSOCOV_CPU_LOCK_ROOT}/${run_id}"
    [[ -d "${DSLASSOCOV_CPU_LOCK_ROOT}" && ! -L "${DSLASSOCOV_CPU_LOCK_ROOT}" ]] || {
        echo "CPU lock root is missing or unsafe: ${DSLASSOCOV_CPU_LOCK_ROOT}" >&2
        return 2
    }
    [[ -d "${lock_dir}" && ! -L "${lock_dir}" ]] || {
        echo "CPU RUN_ID lock directory is missing or unsafe: ${lock_dir}" >&2
        return 2
    }
    [[ "$(realpath -e -- "${DSLASSOCOV_CPU_LOCK_ROOT}")" == "${DSLASSOCOV_CPU_LOCK_ROOT}" ]] || {
        echo "CPU lock root may not resolve through a symlink" >&2
        return 2
    }
    [[ "$(realpath -e -- "${lock_dir}")" == "${lock_dir}" ]] || {
        echo "CPU RUN_ID lock directory may not resolve through a symlink" >&2
        return 2
    }
}

_dslassocov_acquire_cpu_lock() {
    local run_id="$1"
    local lock_name="$2"
    local owner="$3"
    local lock_dir lock_path
    command -v flock >/dev/null 2>&1 || {
        echo "Required CPU lock command is unavailable: flock" >&2
        return 2
    }
    _dslassocov_validate_lock_run_id "${run_id}" || return $?
    _dslassocov_validate_lock_dir "${run_id}" || return $?
    [[ "${lock_name}" =~ ^(launch|seed_[0-9]+)[.]lock$ ]] || {
        echo "Unsafe CPU lock filename: ${lock_name}" >&2
        return 2
    }
    lock_dir="${DSLASSOCOV_CPU_LOCK_ROOT}/${run_id}"
    lock_path="${lock_dir}/${lock_name}"
    [[ ! -L "${lock_path}" ]] || {
        echo "CPU lock file may not be a symlink: ${lock_path}" >&2
        return 2
    }
    exec {DSLASSOCOV_CPU_LOCK_FD}>>"${lock_path}"
    if ! flock --exclusive --nonblock "${DSLASSOCOV_CPU_LOCK_FD}"; then
        echo "cpu_lock_status=busy path=${lock_path}" >&2
        sed -n '1,40p' "${lock_path}" >&2 || true
        exec {DSLASSOCOV_CPU_LOCK_FD}>&-
        DSLASSOCOV_CPU_LOCK_FD=""
        return "${DSLASSOCOV_LOCK_BUSY_EXIT}"
    fi
    DSLASSOCOV_CPU_LOCK_PATH="${lock_path}"
    printf 'event=acquired time=%s pid=%s owner=%s\n' \
        "$(date -Is)" "$$" "${owner}" >&"${DSLASSOCOV_CPU_LOCK_FD}"
    echo "cpu_lock_status=acquired path=${lock_path} pid=$$ owner=${owner}"
}

dslassocov_acquire_launch_lock() {
    local run_id="$1"
    local owner="$2"
    _dslassocov_acquire_cpu_lock "${run_id}" "launch.lock" "${owner}"
}

dslassocov_acquire_seed_lock() {
    local run_id="$1"
    local seed="$2"
    local owner="$3"
    [[ "${seed}" =~ ^[0-9]+$ ]] || {
        echo "CPU seed lock requires a nonnegative integer seed" >&2
        return 2
    }
    _dslassocov_acquire_cpu_lock "${run_id}" "seed_${seed}.lock" "${owner}"
}

dslassocov_release_cpu_lock() {
    local owner="${1:-unknown}"
    if [[ -n "${DSLASSOCOV_CPU_LOCK_FD}" ]]; then
        printf 'event=released time=%s pid=%s owner=%s\n' \
            "$(date -Is)" "$$" "${owner}" >&"${DSLASSOCOV_CPU_LOCK_FD}" || true
        flock --unlock "${DSLASSOCOV_CPU_LOCK_FD}" || true
        exec {DSLASSOCOV_CPU_LOCK_FD}>&-
        echo "cpu_lock_status=released path=${DSLASSOCOV_CPU_LOCK_PATH} pid=$$ owner=${owner}"
    fi
    DSLASSOCOV_CPU_LOCK_FD=""
    DSLASSOCOV_CPU_LOCK_PATH=""
}
