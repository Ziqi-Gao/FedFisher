#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"

fail() {
    echo "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "$(realpath -e -- "${FEDFISHER_PROJECT_ROOT}")" ]] || fail "Status script must run from ${FEDFISHER_PROJECT_ROOT}"

EFFECT_MODIFIER_VARIANT="${EFFECT_MODIFIER_VARIANT:-control}"
case "${EFFECT_MODIFIER_VARIANT}" in
    control)
        default_run_id="dslassocov_control_300seeds"
        ;;
    site_gaussian_shift)
        default_run_id="dslassocov_site_gaussian_shift_300seeds"
        ;;
    local_permutation)
        default_run_id="dslassocov_local_permutation_300seeds"
        ;;
    site_gaussian_local_permutation)
        default_run_id="dslassocov_site_gaussian_local_permutation_300seeds"
        ;;
    *)
        fail "Unknown EFFECT_MODIFIER_VARIANT=${EFFECT_MODIFIER_VARIANT}"
        ;;
esac

RUN_ID="${RUN_ID:-${default_run_id}}"
PYTHON_BIN="${PYTHON_BIN:-${FEDFISHER_SCRATCH_ROOT}/envs/fedfisher-py39/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py status --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}"

SESSION_PREFIX="fedfisher_em_dslassocov_${RUN_ID}"
echo "active_cpu_sessions:"
screen -ls 2>/dev/null | grep "${SESSION_PREFIX}" || echo "  none"

LOG_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/logs/${RUN_ID}"
echo "cpu_worker_log_tails:"
shopt -s nullglob
logs=("${LOG_DIR}"/worker_*.log)
if (( ${#logs[@]} == 0 )); then
    echo "  no logs yet"
else
    for log in "${logs[@]}"; do
        [[ -f "${log}" && ! -L "${log}" ]] || fail "Unsafe worker log: ${log}"
        echo "==> ${log} <=="
        tail -n 12 "${log}"
    done
fi
