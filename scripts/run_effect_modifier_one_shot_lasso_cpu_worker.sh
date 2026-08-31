#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_DATA_ROOT="/data/del6500/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"
readonly PYTHON_BIN="${FEDFISHER_SCRATCH_ROOT}/envs/fedfisher-py39/bin/python"

fail() {
    echo "$*" >&2
    exit 2
}

[[ $# -eq 4 ]] || fail "Usage: $0 VARIANT SHARD_INDEX SHARD_COUNT TOTAL_WORKERS"
variant="$1"
shard_index_raw="$2"
shard_count_raw="$3"
total_workers="$4"
case "${variant}" in
    control|site_gaussian_shift|local_permutation|site_gaussian_local_permutation) ;;
    *) fail "Unknown effect-modifier variant: ${variant}" ;;
esac
[[ "${shard_index_raw}" =~ ^[0-9]+$ ]] || fail "SHARD_INDEX must be a non-negative integer"
[[ "${shard_count_raw}" =~ ^(8|16)$ ]] || fail "SHARD_COUNT must be exactly 8 or 16"
case "${total_workers}" in
    32|64) ;;
    *) fail "TOTAL_WORKERS must be exactly 32 or 64" ;;
esac
shard_index=$((10#${shard_index_raw}))
shard_count=$((10#${shard_count_raw}))
((shard_index < shard_count)) || fail "SHARD_INDEX must be less than SHARD_COUNT"
((shard_count == total_workers / 4)) || fail "SHARD_COUNT must equal TOTAL_WORKERS / 4"
printf -v shard_tag '%02d' "${shard_index}"
printf -v count_tag '%02d' "${shard_count}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "${FEDFISHER_PROJECT_ROOT}" ]] || fail "Worker must run from ${FEDFISHER_PROJECT_ROOT}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"
source "${SCRIPT_DIR}/effect_modifier_one_shot_cpu_lock.sh"

owner="formal_one_shot_worker;variant=${variant};shard=${shard_tag}of${count_tag};pid=$$"
set +e
one_shot_acquire_shard_lock "${variant}" "${shard_index}" "${shard_count}" "${owner}"
lock_code=$?
set -e
case "${lock_code}" in
    0) ;;
    73) fail "Another formal one-shot worker owns ${variant} shard ${shard_tag}of${count_tag}" ;;
    *) fail "Could not acquire formal one-shot worker lock for ${variant} shard ${shard_tag}of${count_tag}" ;;
esac

ACTIVE_CHILD_PID=""

terminate_active_child() {
    local signal_name="$1"
    if [[ -n "${ACTIVE_CHILD_PID}" ]] && kill -0 "${ACTIVE_CHILD_PID}" 2>/dev/null; then
        echo "worker_signal=${signal_name} terminating_child_pid=${ACTIVE_CHILD_PID}" >&2
        kill -TERM "${ACTIVE_CHILD_PID}" 2>/dev/null || true
        wait "${ACTIVE_CHILD_PID}" 2>/dev/null || true
    fi
    exit 130
}

trap 'one_shot_release_cpu_lock "${owner}"' EXIT
trap 'terminate_active_child HUP' HUP
trap 'terminate_active_child INT' INT
trap 'terminate_active_child TERM' TERM

LOG_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/logs/one_shot_stationarity_polish_sharded_v1_300seeds"
RUNTIME_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/runtime/one_shot_stationarity_polish_sharded_v1_300seeds/${variant}/shard_${shard_tag}_of_${count_tag}"
LOG_FILE="${LOG_DIR}/${variant}_shard_${shard_tag}_of_${count_tag}.log"
mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"
for path in "${LOG_DIR}" "${RUNTIME_DIR}"; do
    [[ -d "${path}" && ! -L "${path}" ]] || fail "Unsafe one-shot scratch path: ${path}"
    [[ "$(realpath -e -- "${path}")" == "${path}" ]] || fail "One-shot scratch path may not resolve through a symlink: ${path}"
done
[[ ! -L "${LOG_FILE}" ]] || fail "Worker log may not be a symlink"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${RUNTIME_DIR}/tmp"
export PYTHONPYCACHEPREFIX="${RUNTIME_DIR}/python_cache"
export XDG_CACHE_HOME="${RUNTIME_DIR}/xdg_cache"
export TORCH_HOME="${RUNTIME_DIR}/torch_cache"
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

exec >>"${LOG_FILE}" 2>&1
echo "one_shot_worker_started_at=$(date -Is) variant=${variant} shard=${shard_tag}of${count_tag} pid=$$"
echo "cpu_threads=1 total_workers=${total_workers} families=logistic,mlp seeds=${shard_index}..299/${shard_count} splits=iid,noniid"

cd "${REPO_ROOT}"
manager=("${PYTHON_BIN}" -B effect_modifier_experiment/manage_one_shot_lasso_cpu_run.py)
runner=("${PYTHON_BIN}" -B effect_modifier_experiment/run_one_shot_lasso_effect_modifier.py)

run_cell() {
    local family="$1"
    local seed="$2"
    local split="$3"
    local model="$4"
    local run_id output_dir status_code runner_code
    run_id="${family}_${variant}_stationarity_polish_sharded_v1_300seeds"
    output_dir="${FEDFISHER_DATA_ROOT}/effect_modifier_experiment/outputs/${run_id}"
    set +e
    "${manager[@]}" cell-status --family "${family}" --variant "${variant}" \
        --seed "${seed}" --split "${split}" --model "${model}"
    status_code=$?
    set -e
    case "${status_code}" in
        0)
            echo "skipping_validated_cell family=${family} seed=${seed} split=${split} model=${model}"
            return 0
            ;;
        3) ;;
        *) fail "Partial or invalid cell refused: family=${family} seed=${seed} split=${split} model=${model}" ;;
    esac

    common=(
        --family "${family}"
        --variant "${variant}"
        --run-id "${run_id}"
        --output-dir "${output_dir}"
        --seed "${seed}"
        --seed-start 0
        --seed-end 299
        --split "${split}"
        --model "${model}"
        --num-sites 10
        --site-min-samples 500
        --site-max-samples 1000
        --test-size 10000
        --covariate-dim 100
        --signal-dim 2
        --signal-strength 2.0
        --main-effect-strength 1.0
        --confounding-strength 1.0
        --outcome-intercept 0.0
        --treatment-prob 0.5
        --propensity-clip 0.05
        --alpha 0.1
        --gaussian-mean-span 0.75
        --gaussian-std-low 0.75
        --gaussian-std-high 1.25
        --intervention-repeats 5
        --intervention-batch-size 1024
        --threads 1
    )
    if [[ "${family}" == "one_shot_local_logistic_lasso_fedavg" ]]; then
        common+=(
            --cv-folds 5
            --lambda-count 30
            --lambda-min-ratio 0.001
            --cv-seed-offset 4000037
            --optimizer-max-iter 2000
            --optimizer-polish-max-iter 50
            --optimizer-tolerance 1e-6
            --optimizer-initial-step 1.0
            --optimizer-backtrack-factor 0.5
            --optimizer-min-step 1e-12
            --standardization-scale-floor 1e-12
        )
    else
        common+=(
            --local-epochs 30
            --batch-size 64
            --learning-rate 0.01
            --momentum 0.9
            --validation-fraction 0.2
            --lambda-ratios 0,1,3,10,30,100,300
            --split-seed-offset 5100037
            --loader-seed-offset 5200037
        )
    fi
    echo "cell_started_at=$(date -Is) family=${family} seed=${seed} split=${split} model=${model}"
    set +e
    "${runner[@]}" "${common[@]}" &
    ACTIVE_CHILD_PID=$!
    wait "${ACTIVE_CHILD_PID}"
    runner_code=$?
    ACTIVE_CHILD_PID=""
    set -e
    ((runner_code == 0)) || return "${runner_code}"
    "${manager[@]}" cell-status --family "${family}" --variant "${variant}" \
        --seed "${seed}" --split "${split}" --model "${model}"
    echo "cell_finished_at=$(date -Is) family=${family} seed=${seed} split=${split} model=${model}"
}

for ((seed = shard_index; seed <= 299; seed += shard_count)); do
    for split in iid noniid; do
        run_cell one_shot_local_logistic_lasso_fedavg "${seed}" "${split}" linear
    done
    for split in iid noniid; do
        run_cell one_shot_mlp_group_lasso_fedavg "${seed}" "${split}" SyntheticMLP
        run_cell one_shot_mlp_group_lasso_fedavg "${seed}" "${split}" SyntheticMLPDeep
    done
    echo "seed_all_families_finished=${seed} at=$(date -Is)"
done

echo "one_shot_worker_finished_at=$(date -Is) variant=${variant} shard=${shard_tag}of${count_tag}"
