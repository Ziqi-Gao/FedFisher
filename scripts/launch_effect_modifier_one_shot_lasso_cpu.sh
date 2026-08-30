#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"
readonly PYTHON_BIN="${FEDFISHER_SCRATCH_ROOT}/envs/fedfisher-py39/bin/python"

fail() {
    echo "$*" >&2
    exit 2
}

for command_name in realpath git screen; do
    command -v "${command_name}" >/dev/null 2>&1 || fail "Required command is unavailable: ${command_name}"
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "${FEDFISHER_PROJECT_ROOT}" ]] || fail "Launcher must run from ${FEDFISHER_PROJECT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"

cd "${REPO_ROOT}"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "Formal launch requires a clean source tree"

families=(one_shot_local_logistic_lasso_fedavg one_shot_mlp_group_lasso_fedavg)
variants=(control site_gaussian_shift local_permutation site_gaussian_local_permutation)
sessions=(ff_osp_ctl300 ff_osp_gau300 ff_osp_loc300 ff_osp_cmb300)
manager=("${PYTHON_BIN}" -B effect_modifier_experiment/manage_one_shot_lasso_cpu_run.py)

for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        "${manager[@]}" init --family "${family}" --variant "${variant}"
    done
done

launched=0
for index in "${!variants[@]}"; do
    variant="${variants[index]}"
    session="${sessions[index]}"
    if screen -ls 2>/dev/null | grep -Eq "[0-9]+[.]${session}[[:space:]]"; then
        echo "already_running session=${session} variant=${variant}"
        continue
    fi
    if screen -ls 2>/dev/null | grep -Eq "[0-9]+[.]${session}[[:space:]].*Dead"; then
        fail "Owned one-shot session name exists as Dead; refusing launch without manual review: ${session}"
    fi
    screen -dmS "${session}" "${SCRIPT_DIR}/run_effect_modifier_one_shot_lasso_cpu_worker.sh" "${variant}"
    echo "launched_cpu session=${session} variant=${variant} threads=1"
    launched=$((launched + 1))
done

echo "launched_count=${launched} total_configured_cpu_threads=4"
echo "training_communication_rounds=1 evaluation_communication_separate=1"
echo "status_command=bash scripts/status_effect_modifier_one_shot_lasso_cpu.sh"
