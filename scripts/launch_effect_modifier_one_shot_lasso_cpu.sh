#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"
readonly PYTHON_BIN="${FEDFISHER_SCRATCH_ROOT}/envs/fedfisher-py39/bin/python"
readonly TOTAL_WORKERS="${TOTAL_WORKERS:-32}"

fail() {
    echo "$*" >&2
    exit 2
}

for command_name in realpath git screen lscpu nproc pgrep; do
    command -v "${command_name}" >/dev/null 2>&1 || fail "Required command is unavailable: ${command_name}"
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "${FEDFISHER_PROJECT_ROOT}" ]] || fail "Launcher must run from ${FEDFISHER_PROJECT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"

cd "${REPO_ROOT}"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || fail "Formal launch requires a clean source tree"
case "${TOTAL_WORKERS}" in
    32|64) ;;
    *) fail "TOTAL_WORKERS must be exactly 32 or 64" ;;
esac
readonly WORKERS_PER_VARIANT=$((TOTAL_WORKERS / 4))
readonly PHYSICAL_CORES="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 ":" $2}' | sort -u | wc -l)"
readonly AVAILABLE_CPUS="$(nproc)"
((TOTAL_WORKERS <= PHYSICAL_CORES)) || fail "TOTAL_WORKERS=${TOTAL_WORKERS} exceeds physical cores=${PHYSICAL_CORES}"
((TOTAL_WORKERS <= AVAILABLE_CPUS)) || fail "TOTAL_WORKERS=${TOTAL_WORKERS} exceeds available CPUs=${AVAILABLE_CPUS}"

families=(one_shot_local_logistic_lasso_fedavg one_shot_mlp_group_lasso_fedavg)
variants=(control site_gaussian_shift local_permutation site_gaussian_local_permutation)
variant_tags=(ctl gau loc cmb)
manager=("${PYTHON_BIN}" -B effect_modifier_experiment/manage_one_shot_lasso_cpu_run.py)

for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        "${manager[@]}" init --family "${family}" --variant "${variant}"
    done
done

launched=0
screen_snapshot="$(screen -ls 2>/dev/null || true)"
owned_worker_snapshot="$(pgrep -af '/scripts/run_effect_modifier_one_shot_lasso_cpu_worker[.]sh (control|site_gaussian_shift|local_permutation|site_gaussian_local_permutation) [0-9]+ (8|16) (32|64)$' || true)"
if [[ -n "${owned_worker_snapshot}" ]] &&
   grep -Evq " ${WORKERS_PER_VARIANT} ${TOTAL_WORKERS}$" <<<"${owned_worker_snapshot}"; then
    echo "${owned_worker_snapshot}" >&2
    fail "A different 32/64 one-shot worker layout is still active"
fi
for legacy_session in ff_osp_ctl300 ff_osp_gau300 ff_osp_loc300 ff_osp_cmb300; do
    if grep -Eq "[0-9]+[.]${legacy_session}[[:space:]]" <<<"${screen_snapshot}"; then
        fail "Legacy one-shot session is still present: ${legacy_session}"
    fi
done
for index in "${!variants[@]}"; do
    variant="${variants[index]}"
    variant_tag="${variant_tags[index]}"
    for ((shard_index = 0; shard_index < WORKERS_PER_VARIANT; shard_index += 1)); do
        printf -v shard_tag '%02d' "${shard_index}"
        printf -v count_tag '%02d' "${WORKERS_PER_VARIANT}"
        session="ff_os2_${variant_tag}_${shard_tag}of${count_tag}"
        if screen -ls 2>/dev/null | grep -Eq "[0-9]+[.]${session}[[:space:]].*Dead"; then
            fail "Owned one-shot session name exists as Dead; refusing launch without manual review: ${session}"
        fi
        if screen -ls 2>/dev/null | grep -Eq "[0-9]+[.]${session}[[:space:]]"; then
            echo "already_running session=${session} variant=${variant} shard=${shard_index}/${WORKERS_PER_VARIANT}"
            continue
        fi
        screen -dmS "${session}" "${SCRIPT_DIR}/run_effect_modifier_one_shot_lasso_cpu_worker.sh" "${variant}" "${shard_index}" "${WORKERS_PER_VARIANT}" "${TOTAL_WORKERS}"
        echo "launched_cpu session=${session} variant=${variant} shard=${shard_index}/${WORKERS_PER_VARIANT} threads=1"
        launched=$((launched + 1))
    done
done

echo "launched_count=${launched} total_configured_cpu_threads=${TOTAL_WORKERS}"
echo "workers_per_variant=${WORKERS_PER_VARIANT} physical_cores=${PHYSICAL_CORES} available_cpus=${AVAILABLE_CPUS}"
echo "training_communication_rounds=1 evaluation_communication_separate=1"
echo "status_command=TOTAL_WORKERS=${TOTAL_WORKERS} bash scripts/status_effect_modifier_one_shot_lasso_cpu.sh"
