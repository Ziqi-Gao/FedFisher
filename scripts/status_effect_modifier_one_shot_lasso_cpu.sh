#!/usr/bin/env bash
set -u

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"
readonly PYTHON_BIN="${FEDFISHER_SCRATCH_ROOT}/envs/fedfisher-py39/bin/python"

fail() {
    echo "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "${FEDFISHER_PROJECT_ROOT}" ]] || fail "Status must run from ${FEDFISHER_PROJECT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"
cd "${REPO_ROOT}"

families=(one_shot_local_logistic_lasso_fedavg one_shot_mlp_group_lasso_fedavg)
variants=(control site_gaussian_shift local_permutation site_gaussian_local_permutation)
sessions=(ff_os_ctl300 ff_os_gau300 ff_os_loc300 ff_os_cmb300)
manager=("${PYTHON_BIN}" -B effect_modifier_experiment/manage_one_shot_lasso_cpu_run.py)
status_failures=0

for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        echo "== ${family} / ${variant} =="
        "${manager[@]}" status --family "${family}" --variant "${variant}" || status_failures=$((status_failures + 1))
    done
done

echo "owned_cpu_sessions:"
for session in "${sessions[@]}"; do
    screen -ls 2>/dev/null | grep -E "[0-9]+[.]${session}[[:space:]]" || echo "  absent ${session}"
done

echo "owned_worker_processes:"
pgrep -af '/scripts/run_effect_modifier_one_shot_lasso_cpu_worker[.]sh (control|site_gaussian_shift|local_permutation|site_gaussian_local_permutation)$' || echo "  none"

LOG_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/logs/one_shot_300seeds"
echo "owned_worker_log_tails:"
for variant in "${variants[@]}"; do
    log="${LOG_DIR}/${variant}.log"
    if [[ -f "${log}" && ! -L "${log}" ]]; then
        echo "==> ${log} <=="
        tail -n 10 "${log}"
    else
        echo "  no safe log yet for ${variant}"
    fi
done
echo "status_validation_failures=${status_failures}"
exit "${status_failures}"
