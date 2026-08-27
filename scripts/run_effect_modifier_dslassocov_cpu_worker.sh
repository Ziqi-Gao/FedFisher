#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_DATA_ROOT="/data/del6500/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"

fail() {
    echo "$*" >&2
    exit 2
}

[[ $# -eq 2 ]] || fail "Usage: $0 WORKER_ID WORKER_COUNT"
worker_id="$1"
worker_count="$2"
[[ "${worker_id}" =~ ^[0-9]+$ && "${worker_count}" =~ ^[0-9]+$ ]] || fail "Worker IDs must be nonnegative integers"
(( worker_count >= 1 && worker_id < worker_count )) || fail "Require WORKER_COUNT >= 1 and WORKER_ID < WORKER_COUNT"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "$(realpath -e -- "${FEDFISHER_PROJECT_ROOT}")" ]] || fail "Worker must run from ${FEDFISHER_PROJECT_ROOT}"
source "${SCRIPT_DIR}/effect_modifier_dslassocov_cpu_lock.sh"

: "${RUN_ID:?RUN_ID must be exported by the CPU launcher}"
: "${EFFECT_MODIFIER_VARIANT:?EFFECT_MODIFIER_VARIANT must be exported}"
: "${PYTHON_BIN:?PYTHON_BIN must be exported}"
: "${THREADS_PER_WORKER:?THREADS_PER_WORKER must be exported}"
: "${CPU_WORKERS:?CPU_WORKERS must be exported}"
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"
[[ "${CPU_WORKERS}" =~ ^[1-4]$ ]] || fail "CPU_WORKERS must be in [1, 4]"
[[ "${THREADS_PER_WORKER}" =~ ^[1-4]$ ]] || fail "THREADS_PER_WORKER must be in [1, 4]"
(( worker_count == CPU_WORKERS )) || fail "WORKER_COUNT must equal configured CPU_WORKERS"
(( CPU_WORKERS * THREADS_PER_WORKER <= 4 )) || fail "CPU_WORKERS * THREADS_PER_WORKER must be <= 4"

OUTPUT_DIR="${FEDFISHER_DATA_ROOT}/effect_modifier_experiment/outputs/${RUN_ID}"
LOG_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/logs/${RUN_ID}"
RUNTIME_DIR="${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/runtime/${RUN_ID}/worker_${worker_id}"
LOG_FILE="${LOG_DIR}/worker_${worker_id}.log"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"
[[ "$(realpath -m -- "${OUTPUT_DIR}")" == "${OUTPUT_DIR}" ]] || fail "Output path may not resolve through a symlink"
[[ "$(realpath -m -- "${LOG_DIR}")" == "${LOG_DIR}" ]] || fail "Log path may not resolve through a symlink"
[[ "$(realpath -m -- "${RUNTIME_DIR}")" == "${RUNTIME_DIR}" ]] || fail "Runtime path may not resolve through a symlink"
[[ ! -L "${LOG_FILE}" ]] || fail "Worker log may not be a symlink"

export TMPDIR="${RUNTIME_DIR}/tmp"
export PYTHONPYCACHEPREFIX="${RUNTIME_DIR}/python_cache"
export XDG_CACHE_HOME="${RUNTIME_DIR}/xdg_cache"
export TORCH_HOME="${RUNTIME_DIR}/torch_cache"
mkdir -p "${TMPDIR}" "${PYTHONPYCACHEPREFIX}" "${XDG_CACHE_HOME}" "${TORCH_HOME}"

exec >> "${LOG_FILE}" 2>&1
echo "cpu_worker_started_at=$(date -Is) worker=${worker_id}/${worker_count}"
echo "run_id=${RUN_ID} variant=${EFFECT_MODIFIER_VARIANT}"
echo "threads_per_worker=${THREADS_PER_WORKER}"

cd "${REPO_ROOT}"
IFS=',' read -r -a intervention_modes <<< "${PREDICTION_INTERVENTION_MODES}"
runner_common=(
    --run-id "${RUN_ID}"
    --variant "${EFFECT_MODIFIER_VARIANT}"
    --output-dir "${OUTPUT_DIR}"
    --alpha "${SYNTHETIC_ALPHA}"
    --num-sites "${EFFECT_MODIFIER_NUM_SITES}"
    --num-test "${SYNTHETIC_NUM_TEST}"
    --covariate-dim "${EFFECT_MODIFIER_COVARIATE_DIM}"
    --signal-dim "${EFFECT_MODIFIER_SIGNAL_DIM}"
    --signal-strength "${EFFECT_MODIFIER_SIGNAL_STRENGTH}"
    --main-effect-strength "${EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH}"
    --confounding-strength "${EFFECT_MODIFIER_CONFOUNDING_STRENGTH}"
    --outcome-intercept "${EFFECT_MODIFIER_INTERCEPT}"
    --treatment-prob "${EFFECT_MODIFIER_TREATMENT_PROB}"
    --propensity-clip "${EFFECT_MODIFIER_PROPENSITY_CLIP}"
    --site-min-samples "${EFFECT_MODIFIER_SITE_MIN_SAMPLES}"
    --site-max-samples "${EFFECT_MODIFIER_SITE_MAX_SAMPLES}"
    --site-gaussian-mean-span "${EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN}"
    --site-gaussian-std-low "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW}"
    --site-gaussian-std-high "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH}"
    --intervention-modes "${intervention_modes[@]}"
    --intervention-repeats "${PREDICTION_INTERVENTION_REPEATS}"
    --intervention-batch-size "${PREDICTION_INTERVENTION_BATCH_SIZE}"
    --cv-folds "${CV_FOLDS}"
    --cv-seed-offset "${CV_SEED_OFFSET}"
    --lambda-count "${LAMBDA_COUNT}"
    --lambda-min-ratio "${LAMBDA_MIN_RATIO}"
    --standardization-scale-floor "${STANDARDIZATION_SCALE_FLOOR}"
    --optimizer-max-iter "${OPTIMIZER_MAX_ITER}"
    --optimizer-tolerance "${OPTIMIZER_TOLERANCE}"
    --optimizer-initial-step "${OPTIMIZER_INITIAL_STEP}"
    --optimizer-backtrack-factor "${OPTIMIZER_BACKTRACK_FACTOR}"
    --optimizer-min-step "${OPTIMIZER_MIN_STEP}"
    --threads-per-worker "${THREADS_PER_WORKER}"
    --cpu-workers "${CPU_WORKERS}"
)

first_seed=$((SEED_START + worker_id))
for ((seed = first_seed; seed <= SEED_END; seed += worker_count)); do
    "${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py validate-current --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}"
    seed_owner="worker=${worker_id}/${worker_count};variant=${EFFECT_MODIFIER_VARIANT};seed=${seed};pid=$$"
    set +e
    dslassocov_acquire_seed_lock "${RUN_ID}" "${seed}" "${seed_owner}"
    seed_lock_code=$?
    set -e
    case "${seed_lock_code}" in
        0)
            trap 'dslassocov_release_cpu_lock "${seed_owner}"' EXIT
            trap 'exit 130' HUP INT TERM
            ;;
        73)
            echo "skipping_busy_seed=${seed}"
            continue
            ;;
        *)
            fail "Could not acquire CPU seed lock for seed ${seed}"
            ;;
    esac
    "${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py validate-current --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}"
    set +e
    "${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py seed-status --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}" --seed "${seed}"
    seed_status_code=$?
    set -e
    case "${seed_status_code}" in
        0)
            echo "skipping_validated_seed=${seed}"
            dslassocov_release_cpu_lock "${seed_owner}"
            trap - EXIT HUP INT TERM
            continue
            ;;
        3)
            ;;
        *)
            fail "Seed ${seed} has invalid or partial output; refusing overwrite"
            ;;
    esac

    echo "seed_started=${seed} at=$(date -Is)"
    for split in iid noniid; do
        "${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py validate-current --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}"
        "${PYTHON_BIN}" -B effect_modifier_experiment/run_dslassocov_effect_modifier.py --seed "${seed}" --split "${split}" "${runner_common[@]}"
    done
    "${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py complete-seed --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}" --seed "${seed}"
    echo "seed_finished=${seed} at=$(date -Is)"
    dslassocov_release_cpu_lock "${seed_owner}"
    trap - EXIT HUP INT TERM
done

echo "cpu_worker_finished_at=$(date -Is)"
