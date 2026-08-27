#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_DATA_ROOT="/data/del6500/FedFisher"
readonly FEDFISHER_SCRATCH_ROOT="/scr/del6500/FedFisher"

fail() {
    echo "$*" >&2
    exit 2
}

for required_command in realpath git flock; do
    command -v "${required_command}" >/dev/null 2>&1 || fail "Required command is unavailable: ${required_command}"
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
[[ "$(realpath -e -- "${REPO_ROOT}")" == "$(realpath -e -- "${FEDFISHER_PROJECT_ROOT}")" ]] || fail "Launcher must run from ${FEDFISHER_PROJECT_ROOT}"
source "${SCRIPT_DIR}/effect_modifier_dslassocov_cpu_lock.sh"

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
CPU_WORKERS="${CPU_WORKERS:-2}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"
RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-0}"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-299}"
SPLITS="${SPLITS:-iid,noniid}"
SYNTHETIC_ALPHA="${SYNTHETIC_ALPHA:-0.1}"
EFFECT_MODIFIER_NUM_SITES="${EFFECT_MODIFIER_NUM_SITES:-10}"
SYNTHETIC_NUM_TEST="${SYNTHETIC_NUM_TEST:-10000}"
EFFECT_MODIFIER_COVARIATE_DIM="${EFFECT_MODIFIER_COVARIATE_DIM:-100}"
EFFECT_MODIFIER_SIGNAL_DIM="${EFFECT_MODIFIER_SIGNAL_DIM:-2}"
EFFECT_MODIFIER_SIGNAL_STRENGTH="${EFFECT_MODIFIER_SIGNAL_STRENGTH:-2.0}"
EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH="${EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH:-1.0}"
EFFECT_MODIFIER_CONFOUNDING_STRENGTH="${EFFECT_MODIFIER_CONFOUNDING_STRENGTH:-1.0}"
EFFECT_MODIFIER_INTERCEPT="${EFFECT_MODIFIER_INTERCEPT:-0.0}"
EFFECT_MODIFIER_TREATMENT_PROB="${EFFECT_MODIFIER_TREATMENT_PROB:-0.5}"
EFFECT_MODIFIER_PROPENSITY_CLIP="${EFFECT_MODIFIER_PROPENSITY_CLIP:-0.05}"
EFFECT_MODIFIER_SITE_MIN_SAMPLES="${EFFECT_MODIFIER_SITE_MIN_SAMPLES:-500}"
EFFECT_MODIFIER_SITE_MAX_SAMPLES="${EFFECT_MODIFIER_SITE_MAX_SAMPLES:-1000}"
EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN="${EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN:-0.75}"
EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW="${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW:-0.75}"
EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH="${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH:-1.25}"
PREDICTION_INTERVENTION_MODES="${PREDICTION_INTERVENTION_MODES:-permute,zero}"
PREDICTION_INTERVENTION_REPEATS="${PREDICTION_INTERVENTION_REPEATS:-5}"
PREDICTION_INTERVENTION_BATCH_SIZE="${PREDICTION_INTERVENTION_BATCH_SIZE:-1024}"
CV_FOLDS="${CV_FOLDS:-5}"
CV_SEED_OFFSET="${CV_SEED_OFFSET:-4000037}"
LAMBDA_COUNT="${LAMBDA_COUNT:-30}"
LAMBDA_MIN_RATIO="${LAMBDA_MIN_RATIO:-0.001}"
STANDARDIZATION_SCALE_FLOOR="${STANDARDIZATION_SCALE_FLOOR:-1e-12}"
OPTIMIZER_MAX_ITER="${OPTIMIZER_MAX_ITER:-2000}"
OPTIMIZER_TOLERANCE="${OPTIMIZER_TOLERANCE:-1e-6}"
OPTIMIZER_INITIAL_STEP="${OPTIMIZER_INITIAL_STEP:-1.0}"
OPTIMIZER_BACKTRACK_FACTOR="${OPTIMIZER_BACKTRACK_FACTOR:-0.5}"
OPTIMIZER_MIN_STEP="${OPTIMIZER_MIN_STEP:-1e-12}"

for value in "${CPU_WORKERS}" "${THREADS_PER_WORKER}" "${RUN_IN_FOREGROUND}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "CPU worker controls must be integers"
done
(( CPU_WORKERS >= 1 && CPU_WORKERS <= 4 )) || fail "CPU_WORKERS must be in [1, 4]"
(( THREADS_PER_WORKER >= 1 && THREADS_PER_WORKER <= 4 )) || fail "THREADS_PER_WORKER must be in [1, 4]"
(( CPU_WORKERS * THREADS_PER_WORKER <= 4 )) || fail "CPU_WORKERS * THREADS_PER_WORKER must be <= 4"
[[ "${RUN_IN_FOREGROUND}" == "0" || "${RUN_IN_FOREGROUND}" == "1" ]] || fail "RUN_IN_FOREGROUND must be 0 or 1"
if [[ "${RUN_IN_FOREGROUND}" == "1" && "${CPU_WORKERS}" != "1" ]]; then
    fail "RUN_IN_FOREGROUND=1 requires CPU_WORKERS=1"
fi
[[ -x "${PYTHON_BIN}" ]] || fail "Python environment is unavailable: ${PYTHON_BIN}"
if [[ "${RUN_IN_FOREGROUND}" == "0" ]]; then
    command -v screen >/dev/null 2>&1 || fail "GNU Screen is required for detached launch"
fi

export EFFECT_MODIFIER_VARIANT RUN_ID PYTHON_BIN CPU_WORKERS THREADS_PER_WORKER
export SEED_START SEED_END SPLITS SYNTHETIC_ALPHA EFFECT_MODIFIER_NUM_SITES
export SYNTHETIC_NUM_TEST EFFECT_MODIFIER_COVARIATE_DIM EFFECT_MODIFIER_SIGNAL_DIM
export EFFECT_MODIFIER_SIGNAL_STRENGTH EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH
export EFFECT_MODIFIER_CONFOUNDING_STRENGTH EFFECT_MODIFIER_INTERCEPT
export EFFECT_MODIFIER_TREATMENT_PROB EFFECT_MODIFIER_PROPENSITY_CLIP
export EFFECT_MODIFIER_SITE_MIN_SAMPLES EFFECT_MODIFIER_SITE_MAX_SAMPLES
export EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN
export EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH
export PREDICTION_INTERVENTION_MODES PREDICTION_INTERVENTION_REPEATS
export PREDICTION_INTERVENTION_BATCH_SIZE CV_FOLDS CV_SEED_OFFSET
export LAMBDA_COUNT LAMBDA_MIN_RATIO STANDARDIZATION_SCALE_FLOOR
export OPTIMIZER_MAX_ITER OPTIMIZER_TOLERANCE OPTIMIZER_INITIAL_STEP
export OPTIMIZER_BACKTRACK_FACTOR OPTIMIZER_MIN_STEP

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"
export PYTHONDONTWRITEBYTECODE=1

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -B effect_modifier_experiment/manage_dslassocov_cpu_run.py init --run-id "${RUN_ID}" --variant "${EFFECT_MODIFIER_VARIANT}"
launch_owner="launcher;variant=${EFFECT_MODIFIER_VARIANT};pid=$$"
set +e
dslassocov_acquire_launch_lock "${RUN_ID}" "${launch_owner}"
launch_lock_code=$?
set -e
case "${launch_lock_code}" in
    0) ;;
    73) fail "Another launcher owns RUN_ID ${RUN_ID}; refusing concurrent launch" ;;
    *) fail "Could not acquire CPU launch lock for RUN_ID ${RUN_ID}" ;;
esac
trap 'dslassocov_release_cpu_lock "${launch_owner}"' EXIT
trap 'exit 130' HUP INT TERM

SESSION_PREFIX="fedfisher_em_dslassocov_${RUN_ID}"
if [[ "${RUN_IN_FOREGROUND}" == "1" ]]; then
    "${SCRIPT_DIR}/run_effect_modifier_dslassocov_cpu_worker.sh" 0 1
else
    launched=0
    for ((worker_id = 0; worker_id < CPU_WORKERS; worker_id += 1)); do
        session="${SESSION_PREFIX}_w${worker_id}"
        if screen -ls 2>/dev/null | grep -Eq "[0-9]+[.]${session}[[:space:]]"; then
            echo "already_running session=${session}"
            continue
        fi
        # Screen and every descendant must not inherit the launcher's lock FD.
        screen -dmS "${session}" "${SCRIPT_DIR}/run_effect_modifier_dslassocov_cpu_worker.sh" \
            "${worker_id}" "${CPU_WORKERS}" \
            {DSLASSOCOV_CPU_LOCK_FD}>&-
        echo "launched_cpu session=${session} worker=${worker_id}"
        launched=$((launched + 1))
    done
    echo "launched_count=${launched}"
fi
dslassocov_release_cpu_lock "${launch_owner}"
trap - EXIT HUP INT TERM


echo "cpu_workers=${CPU_WORKERS} threads_per_worker=${THREADS_PER_WORKER}"
echo "run_id=${RUN_ID} variant=${EFFECT_MODIFIER_VARIANT}"
echo "output_dir=${FEDFISHER_DATA_ROOT}/effect_modifier_experiment/outputs/${RUN_ID}"
echo "log_dir=${FEDFISHER_SCRATCH_ROOT}/effect_modifier_experiment/logs/${RUN_ID}"
echo "status_command=EFFECT_MODIFIER_VARIANT=${EFFECT_MODIFIER_VARIANT} RUN_ID=${RUN_ID} bash scripts/status_effect_modifier_dslassocov_cpu.sh"
