#!/usr/bin/env bash
set -u

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"
readonly PYTHON_BIN="${FEDFISHER_SCRATCH_ROOT}/envs/fedfisher-py39/bin/python"
readonly TOTAL_WORKERS="${TOTAL_WORKERS:-32}"

fail() {
    echo "$*" >&2
    exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "${FEDFISHER_PROJECT_ROOT}" ]] || fail "Status must run from ${FEDFISHER_PROJECT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"
cd "${REPO_ROOT}"
case "${TOTAL_WORKERS}" in
    32|64) ;;
    *) fail "TOTAL_WORKERS must be exactly 32 or 64" ;;
esac
readonly WORKERS_PER_VARIANT=$((TOTAL_WORKERS / 4))

families=(one_shot_local_logistic_lasso_fedavg one_shot_mlp_group_lasso_fedavg)
variants=(control site_gaussian_shift local_permutation site_gaussian_local_permutation)
variant_tags=(ctl gau loc cmb)
manager=("${PYTHON_BIN}" -B effect_modifier_experiment/manage_one_shot_lasso_cpu_run.py)
status_failures=0

for family in "${families[@]}"; do
    for variant in "${variants[@]}"; do
        echo "== ${family} / ${variant} =="
        "${manager[@]}" status --family "${family}" --variant "${variant}" || status_failures=$((status_failures + 1))
    done
done

echo "owned_cpu_sessions:"
screen_snapshot="$(screen -ls 2>/dev/null || true)"
owned_sessions_present=0
for index in "${!variants[@]}"; do
    variant_tag="${variant_tags[index]}"
    for ((shard_index = 0; shard_index < WORKERS_PER_VARIANT; shard_index += 1)); do
        printf -v shard_tag '%02d' "${shard_index}"
        printf -v count_tag '%02d' "${WORKERS_PER_VARIANT}"
        session="ff_os2_${variant_tag}_${shard_tag}of${count_tag}"
        if grep -E "[0-9]+[.]${session}[[:space:]]" <<<"${screen_snapshot}"; then
            owned_sessions_present=$((owned_sessions_present + 1))
        else
            echo "  absent ${session}"
        fi
    done
done
echo "owned_sessions_present=${owned_sessions_present} expected=${TOTAL_WORKERS}"

echo "owned_worker_processes:"
pgrep -af '/scripts/run_effect_modifier_one_shot_lasso_cpu_worker[.]sh (control|site_gaussian_shift|local_permutation|site_gaussian_local_permutation) [0-9]+ (8|16) (32|64)$' || echo "  none"

LOG_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/logs/one_shot_stationarity_polish_sharded_v1_300seeds"
echo "owned_worker_log_tails:"
for variant in "${variants[@]}"; do
    printf -v count_tag '%02d' "${WORKERS_PER_VARIANT}"
    latest_log="$(find "${LOG_DIR}" -maxdepth 1 -type f -name "${variant}_shard_*_of_${count_tag}.log" -printf '%T@ %p\n' 2>/dev/null | sort -nr | sed -n '1s/^[^ ]* //p')"
    if [[ -n "${latest_log}" && -f "${latest_log}" && ! -L "${latest_log}" ]]; then
        echo "==> latest ${latest_log} <=="
        tail -n 10 "${latest_log}"
    else
        echo "  no safe log yet for ${variant}"
    fi
done
echo "configured_total_workers=${TOTAL_WORKERS} workers_per_variant=${WORKERS_PER_VARIANT}"
echo "status_validation_failures=${status_failures}"
exit "${status_failures}"
