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

    [[ "${candidate}" == /* ]] || fail "${label} must be an absolute path: ${candidate}"
    resolved_candidate="$(realpath -m -- "${candidate}")"
    resolved_root="$(realpath -e -- "${allowed_root}")"
    expected="${resolved_root}/${relative_path}"
    [[ "${resolved_candidate}" == "${expected}" ]] \
        || fail "${label} must resolve to ${expected}: ${resolved_candidate}"
    printf '%s\n' "${resolved_candidate}"
}

validate_run_id_namespace() {
    case "${EFFECT_MODIFIER_VARIANT}" in
        control)
            [[ "${RUN_ID}" == prediction_intervention_* ]] \
                || fail "control RUN_ID must start with prediction_intervention_"
            ;;
        site_gaussian_shift)
            [[ "${RUN_ID}" == site_gaussian_shift_* ]] \
                || fail "site_gaussian_shift RUN_ID must start with site_gaussian_shift_"
            ;;
        local_permutation)
            [[ "${RUN_ID}" == local_permutation_* ]]                 || fail "local_permutation RUN_ID must start with local_permutation_"
            ;;
    esac
}

if [[ $# -ne 3 ]]; then
    fail "Usage: $0 WORKER_ID WORKER_COUNT GPU_ID"
fi

worker_id="$1"
worker_count="$2"
gpu_id="$3"
for value in "${worker_id}" "${worker_count}" "${gpu_id}"; do
    [[ "${value}" =~ ^[0-9]+$ ]]         || fail "Worker identifiers and GPU ID must be non-negative integers"
done
(( worker_count >= 1 && worker_id < worker_count ))     || fail "Require WORKER_COUNT >= 1 and WORKER_ID < WORKER_COUNT"

command -v realpath >/dev/null 2>&1 || fail "Required command is unavailable: realpath"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
resolved_repo="$(realpath -e -- "${SCRIPT_DIR}/..")"
[[ "${resolved_repo}" == "$(realpath -e -- "${FEDFISHER_PROJECT_ROOT}")" ]] \
    || fail "Slot script must resolve inside ${FEDFISHER_PROJECT_ROOT}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${FEDFISHER_SCRATCH_ROOT}}"
EFFECT_MODIFIER_VARIANT="${EFFECT_MODIFIER_VARIANT:-control}"
case "${EFFECT_MODIFIER_VARIANT}" in
    control)
        default_run_id="prediction_intervention_ablation_control_300seeds"
        ;;
    site_gaussian_shift)
        default_run_id="site_gaussian_shift_300seeds"
        ;;
    local_permutation)
        default_run_id="local_permutation_300seeds"
        ;;
    *)
        fail "EFFECT_MODIFIER_VARIANT must be control, site_gaussian_shift, or local_permutation"
        ;;
esac
RUN_ID="${RUN_ID:-${default_run_id}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]     || fail "RUN_ID may only contain letters, digits, dot, underscore, and hyphen"
validate_run_id_namespace

SCRATCH_ROOT="$(canonical_exact "SCRATCH_ROOT" "${SCRATCH_ROOT}" "${FEDFISHER_SCRATCH_ROOT}")"
LOG_DIR="$(canonical_bound_path \
    "LOG_DIR" \
    "${SCRATCH_ROOT}/effect_modifier_experiment/logs/${RUN_ID}" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "effect_modifier_experiment/logs/${RUN_ID}")"
LOCK_DIR="$(canonical_bound_path \
    "LOCK_DIR" \
    "${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/locks" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "effect_modifier_experiment/locks")"
LOG_FILE="${LOG_DIR}/worker_${worker_id}_gpu_${gpu_id}.log"
LOCK_FILE="${LOCK_DIR}/gpu_${gpu_id}.lock"
WAIT_FOR_GPU_IDLE="${WAIT_FOR_GPU_IDLE:-1}"
GPU_IDLE_POLLS="${GPU_IDLE_POLLS:-3}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

for value in "${WAIT_FOR_GPU_IDLE}" "${GPU_IDLE_POLLS}" "${GPU_POLL_SECONDS}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "GPU wait controls must be non-negative integers"
done
[[ "${WAIT_FOR_GPU_IDLE}" == "0" || "${WAIT_FOR_GPU_IDLE}" == "1" ]]     || fail "WAIT_FOR_GPU_IDLE must be 0 or 1"
(( GPU_IDLE_POLLS >= 1 )) || fail "GPU_IDLE_POLLS must be at least 1"

mkdir -p "${LOG_DIR}" "${LOCK_DIR}"
LOG_DIR="$(canonical_bound_path "LOG_DIR" "${LOG_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "effect_modifier_experiment/logs/${RUN_ID}")"
LOCK_DIR="$(canonical_bound_path "LOCK_DIR" "${LOCK_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "effect_modifier_experiment/locks")"
[[ ! -L "${LOG_FILE}" ]] || fail "Worker log may not be a symlink: ${LOG_FILE}"
[[ ! -L "${LOCK_FILE}" ]] || fail "GPU lock may not be a symlink: ${LOCK_FILE}"

exec >> "${LOG_FILE}" 2>&1

echo "slot_started_at=$(date -Is) variant=${EFFECT_MODIFIER_VARIANT} worker_id=${worker_id} gpu_id=${gpu_id}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another FedFisher slot already owns GPU ${gpu_id}; exiting" >&2
    exit 1
fi

if [[ "${WAIT_FOR_GPU_IDLE}" == "1" ]]; then
    idle_count=0
    last_pids=""
    while (( idle_count < GPU_IDLE_POLLS )); do
        pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')"
        if [[ -z "${pids}" ]]; then
            idle_count=$((idle_count + 1))
            echo "gpu=${gpu_id} idle_poll=${idle_count}/${GPU_IDLE_POLLS} at=$(date -Is)"
        else
            idle_count=0
            if [[ "${pids}" != "${last_pids}" ]]; then
                echo "gpu=${gpu_id} waiting_for_pids=${pids} at=$(date -Is)"
            fi
        fi
        last_pids="${pids}"
        if (( idle_count < GPU_IDLE_POLLS )); then
            sleep "${GPU_POLL_SECONDS}"
        fi
    done
fi

echo "gpu=${gpu_id} acquired_at=$(date -Is)"
exec "${SCRIPT_DIR}/run_synthetic_effect_modifier_local_worker.sh"     "${worker_id}" "${worker_count}" "${gpu_id}"
