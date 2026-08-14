#!/usr/bin/env bash
set -euo pipefail

readonly FEDFISHER_PROJECT_ROOT="/home/del6500/projects/FedFisher"
readonly FEDFISHER_DATA_ROOT="/data/del6500/FedFisher"
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
    [[ "${resolved_candidate}" == "${resolved_expected}" ]]         || fail "${label} must resolve to ${expected}: ${resolved_candidate}"
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

metadata_value() {
    local path="$1"
    local key="$2"
    awk -F= -v expected_key="${key}" '
        $1 == expected_key {
            count += 1
            print substr($0, length(expected_key) + 2)
        }
        END {
            if (count != 1) {
                exit 1
            }
        }
    ' "${path}"
}

compute_source_tree_sha256() {
    (
        cd "${REPO_ROOT}"
        git ls-files --cached --others --exclude-standard -z \
            | while IFS= read -r -d '' source_path; do
                if [[ -e "${source_path}" || -L "${source_path}" ]]; then
                    printf '%s\0' "${source_path}"
                fi
            done \
            | LC_ALL=C sort -z \
            | tar \
                --null \
                --no-recursion \
                --create \
                --file=- \
                --format=gnu \
                --mtime="@0" \
                --owner=0 \
                --group=0 \
                --numeric-owner \
                --files-from=-
    ) | sha256sum | awk '{print $1}'
}

validate_run_manifest() {
    local manifest="$1"
    local key
    local expected
    local actual

    [[ -f "${manifest}" && ! -L "${manifest}" ]]         || fail "Run manifest must be a regular non-symlink file: ${manifest}"
    while IFS='|' read -r key expected; do
        actual="$(metadata_value "${manifest}" "${key}")" \
            || fail "Run manifest has missing or duplicate key ${key}: ${manifest}"
        [[ "${actual}" == "${expected}" ]]             || fail "Run manifest mismatch for ${key}: expected ${expected}, found ${actual}"
    done <<EOF
format_version|2
run_id|${RUN_ID}
effect_modifier_variant|${EFFECT_MODIFIER_VARIANT}
git_commit|${GIT_COMMIT}
git_dirty|${GIT_DIRTY}
git_status_sha256|${GIT_STATUS_SHA256}
source_tree_sha256|${SOURCE_TREE_SHA256}
provenance_sha256|${RUN_PROVENANCE_SHA256}
run_config_sha256|${RUN_CONFIG_SHA256}
variant_config_sha256|${VARIANT_CONFIG_SHA256}
effect_modifier_site_gaussian_shift|${SITE_GAUSSIAN_SHIFT_ENABLED}
prediction_intervention_scope|${PREDICTION_INTERVENTION_SCOPE}
EOF
}

initialize_or_validate_run_manifest() {
    local manifest="${OUTPUT_DIR}/run_manifest.env"
    local manifest_tmp
    local unexpected_entry

    [[ ! -L "${manifest}" ]] || fail "Run manifest may not be a symlink: ${manifest}"
    if [[ ! -e "${manifest}" ]]; then
        unexpected_entry="$(
            find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit
        )"
        [[ -z "${unexpected_entry}" ]]             || fail "Refusing non-empty output directory without run_manifest.env: ${OUTPUT_DIR}"

        manifest_tmp="${manifest}.tmp.$$"
        [[ ! -e "${manifest_tmp}" && ! -L "${manifest_tmp}" ]] \
            || fail "Temporary manifest path already exists: ${manifest_tmp}"
        (
            umask 077
            printf '%s\n' \
                "format_version=2" \
                "run_id=${RUN_ID}" \
                "effect_modifier_variant=${EFFECT_MODIFIER_VARIANT}" \
                "git_commit=${GIT_COMMIT}" \
                "git_dirty=${GIT_DIRTY}" \
                "git_status_sha256=${GIT_STATUS_SHA256}" \
                "source_tree_sha256=${SOURCE_TREE_SHA256}" \
                "provenance_sha256=${RUN_PROVENANCE_SHA256}" \
                "run_config_sha256=${RUN_CONFIG_SHA256}" \
                "variant_config_sha256=${VARIANT_CONFIG_SHA256}" \
                "${RUN_CONFIG_TEXT}" \
                "${VARIANT_CONFIG_TEXT}" \
                "created_at=$(date -Is)" > "${manifest_tmp}"
        )
        if ! ln -- "${manifest_tmp}" "${manifest}" 2>/dev/null; then
            rm -- "${manifest_tmp}"
            [[ -e "${manifest}" ]]                 || fail "Could not atomically create run manifest: ${manifest}"
        else
            rm -- "${manifest_tmp}"
        fi
    fi
    validate_run_manifest "${manifest}"
}

for required_command in realpath git tar sha256sum awk find ln screen nvidia-smi; do
    command -v "${required_command}" >/dev/null 2>&1         || fail "Required command is unavailable: ${required_command}"
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
DATA_ROOT="${DATA_ROOT:-${FEDFISHER_DATA_ROOT}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-${FEDFISHER_SCRATCH_ROOT}}"
EFFECT_MODIFIER_VARIANT="${EFFECT_MODIFIER_VARIANT:-control}"

case "${EFFECT_MODIFIER_VARIANT}" in
    control)
        default_run_id="prediction_intervention_ablation_control_300seeds"
        SITE_GAUSSIAN_SHIFT_ENABLED=0
        PREDICTION_INTERVENTION_SCOPE=global
        ;;
    site_gaussian_shift)
        default_run_id="site_gaussian_shift_300seeds"
        SITE_GAUSSIAN_SHIFT_ENABLED=1
        PREDICTION_INTERVENTION_SCOPE=global
        ;;
    local_permutation)
        default_run_id="local_permutation_300seeds"
        SITE_GAUSSIAN_SHIFT_ENABLED=0
        PREDICTION_INTERVENTION_SCOPE=site
        ;;
    *)
        fail "EFFECT_MODIFIER_VARIANT must be control, site_gaussian_shift, or local_permutation"
        ;;
esac

RUN_ID="${RUN_ID:-${default_run_id}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/effect_modifier_experiment/outputs/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-${SCRATCH_ROOT}/envs/fedfisher-py39/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SESSION_PREFIX="fedfisher_em_${RUN_ID}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]     || fail "RUN_ID may only contain letters, digits, dot, underscore, and hyphen"
validate_run_id_namespace

REPO_ROOT="$(canonical_exact "REPO_ROOT" "${REPO_ROOT}" "${FEDFISHER_PROJECT_ROOT}")"
DATA_ROOT="$(canonical_exact "DATA_ROOT" "${DATA_ROOT}" "${FEDFISHER_DATA_ROOT}")"
SCRATCH_ROOT="$(canonical_exact "SCRATCH_ROOT" "${SCRATCH_ROOT}" "${FEDFISHER_SCRATCH_ROOT}")"
OUTPUT_DIR="$(canonical_bound_path \
    "OUTPUT_DIR" \
    "${OUTPUT_DIR}" \
    "${FEDFISHER_DATA_ROOT}" \
    "effect_modifier_experiment/outputs/${RUN_ID}")"
LOG_DIR="$(canonical_bound_path \
    "LOG_DIR" \
    "${SCRATCH_ROOT}/effect_modifier_experiment/logs/${RUN_ID}" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "effect_modifier_experiment/logs/${RUN_ID}")"
RUNTIME_DIR="$(canonical_bound_path \
    "RUNTIME_DIR" \
    "${SCRATCH_ROOT}/effect_modifier_experiment/runtime/${RUN_ID}" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "effect_modifier_experiment/runtime/${RUN_ID}")"

SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-299}"
SYNTHETIC_ALPHA="${SYNTHETIC_ALPHA:-0.1}"
EFFECT_MODIFIER_NUM_SITES="${EFFECT_MODIFIER_NUM_SITES:-10}"
SYNTHETIC_LOCAL_EPOCHS="${SYNTHETIC_LOCAL_EPOCHS:-30}"
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
PREDICTION_INTERVENTION_REPEATS="${PREDICTION_INTERVENTION_REPEATS:-5}"
PREDICTION_INTERVENTION_BATCH_SIZE="${PREDICTION_INTERVENTION_BATCH_SIZE:-1024}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

for config_value in \
    "${SEED_START}" \
    "${SEED_END}" \
    "${SYNTHETIC_ALPHA}" \
    "${EFFECT_MODIFIER_NUM_SITES}" \
    "${SYNTHETIC_LOCAL_EPOCHS}" \
    "${SYNTHETIC_NUM_TEST}" \
    "${EFFECT_MODIFIER_COVARIATE_DIM}" \
    "${EFFECT_MODIFIER_SIGNAL_DIM}" \
    "${EFFECT_MODIFIER_SIGNAL_STRENGTH}" \
    "${EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH}" \
    "${EFFECT_MODIFIER_CONFOUNDING_STRENGTH}" \
    "${EFFECT_MODIFIER_INTERCEPT}" \
    "${EFFECT_MODIFIER_TREATMENT_PROB}" \
    "${EFFECT_MODIFIER_PROPENSITY_CLIP}" \
    "${EFFECT_MODIFIER_SITE_MIN_SAMPLES}" \
    "${EFFECT_MODIFIER_SITE_MAX_SAMPLES}" \
    "${EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN}" \
    "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW}" \
    "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH}" \
    "${PREDICTION_INTERVENTION_REPEATS}" \
    "${PREDICTION_INTERVENTION_BATCH_SIZE}" \
    "${OMP_NUM_THREADS}" \
    "${MKL_NUM_THREADS}"; do
    [[ "${config_value}" =~ ^[^=[:space:]]+$ ]] \
        || fail "Run configuration values must be nonempty scalars without whitespace or '='"
done

RUN_CONFIG_TEXT="$(
    printf '%s\n' \
        "seed_start=${SEED_START}" \
        "seed_end=${SEED_END}" \
        "models=SyntheticMLP,SyntheticMLPDeep" \
        "splits=iid,noniid" \
        "algs=fedavg,fedfisher_diag,fedfisher_kfac" \
        "synthetic_alpha=${SYNTHETIC_ALPHA}" \
        "effect_modifier_num_sites=${EFFECT_MODIFIER_NUM_SITES}" \
        "num_rounds=1" \
        "synthetic_local_epochs=${SYNTHETIC_LOCAL_EPOCHS}" \
        "synthetic_num_test=${SYNTHETIC_NUM_TEST}" \
        "effect_modifier_covariate_dim=${EFFECT_MODIFIER_COVARIATE_DIM}" \
        "effect_modifier_signal_dim=${EFFECT_MODIFIER_SIGNAL_DIM}" \
        "effect_modifier_signal_strength=${EFFECT_MODIFIER_SIGNAL_STRENGTH}" \
        "effect_modifier_main_effect_strength=${EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH}" \
        "effect_modifier_confounding_strength=${EFFECT_MODIFIER_CONFOUNDING_STRENGTH}" \
        "effect_modifier_intercept=${EFFECT_MODIFIER_INTERCEPT}" \
        "effect_modifier_treatment_prob=${EFFECT_MODIFIER_TREATMENT_PROB}" \
        "effect_modifier_propensity_clip=${EFFECT_MODIFIER_PROPENSITY_CLIP}" \
        "effect_modifier_site_min_samples=${EFFECT_MODIFIER_SITE_MIN_SAMPLES}" \
        "effect_modifier_site_max_samples=${EFFECT_MODIFIER_SITE_MAX_SAMPLES}" \
        "effect_modifier_site_gaussian_mean_span=${EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN}" \
        "effect_modifier_site_gaussian_std_low=${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW}" \
        "effect_modifier_site_gaussian_std_high=${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH}" \
        "prediction_intervention_modes=permute,zero" \
        "prediction_intervention_repeats=${PREDICTION_INTERVENTION_REPEATS}" \
        "prediction_intervention_batch_size=${PREDICTION_INTERVENTION_BATCH_SIZE}" \
        "omp_num_threads=${OMP_NUM_THREADS}" \
        "mkl_num_threads=${MKL_NUM_THREADS}"
)"
RUN_CONFIG_SHA256="$(
    printf '%s\n' "${RUN_CONFIG_TEXT}" | sha256sum | awk '{print $1}'
)"
VARIANT_CONFIG_TEXT="$(
    printf '%s\n' \
        "effect_modifier_site_gaussian_shift=${SITE_GAUSSIAN_SHIFT_ENABLED}" \
        "prediction_intervention_scope=${PREDICTION_INTERVENTION_SCOPE}"
)"
VARIANT_CONFIG_SHA256="$(
    printf '%s\n' "${VARIANT_CONFIG_TEXT}" | sha256sum | awk '{print $1}'
)"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Environment missing: ${PYTHON_BIN}" >&2
    echo "Run scripts/setup_lab_server_env.sh first" >&2
    exit 1
fi

IFS=',' read -r -a gpu_array <<< "${GPU_IDS}"
worker_count="${#gpu_array[@]}"
(( worker_count >= 1 )) || fail "GPU_IDS must contain at least one GPU"

for gpu_id in "${gpu_array[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]]         || fail "GPU_IDS must be a comma-separated list of non-negative integers"
    nvidia-smi -i "${gpu_id}" --query-gpu=index,name --format=csv,noheader >/dev/null
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${RUNTIME_DIR}"
OUTPUT_DIR="$(canonical_bound_path "OUTPUT_DIR" "${OUTPUT_DIR}" "${FEDFISHER_DATA_ROOT}" "effect_modifier_experiment/outputs/${RUN_ID}")"
LOG_DIR="$(canonical_bound_path "LOG_DIR" "${LOG_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "effect_modifier_experiment/logs/${RUN_ID}")"
RUNTIME_DIR="$(canonical_bound_path "RUNTIME_DIR" "${RUNTIME_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "effect_modifier_experiment/runtime/${RUN_ID}")"

GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)"
GIT_STATUS="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
if [[ -n "${GIT_STATUS}" ]]; then
    GIT_DIRTY=1
else
    GIT_DIRTY=0
fi
GIT_STATUS_SHA256="$(
    printf '%s\n' "${GIT_STATUS}" | sha256sum | awk '{print $1}'
)"
SOURCE_TREE_SHA256="$(compute_source_tree_sha256)"
RUN_PROVENANCE_SHA256="$(
    printf '%s\n' \
        "git_commit=${GIT_COMMIT}" \
        "git_dirty=${GIT_DIRTY}" \
        "git_status_sha256=${GIT_STATUS_SHA256}" \
        "source_tree_sha256=${SOURCE_TREE_SHA256}" \
        | sha256sum \
        | awk '{print $1}'
)"
initialize_or_validate_run_manifest
unsafe_output_symlink="$(find "${OUTPUT_DIR}" -type l -print -quit)"
[[ -z "${unsafe_output_symlink}" ]] \
    || fail "Output tree may not contain symlinks: ${unsafe_output_symlink}"

export REPO_ROOT DATA_ROOT SCRATCH_ROOT EFFECT_MODIFIER_VARIANT RUN_ID OUTPUT_DIR
export PYTHON_BIN RUN_PROVENANCE_SHA256 RUN_CONFIG_SHA256 VARIANT_CONFIG_SHA256
export SEED_START SEED_END SYNTHETIC_ALPHA EFFECT_MODIFIER_NUM_SITES
export SYNTHETIC_LOCAL_EPOCHS SYNTHETIC_NUM_TEST
export EFFECT_MODIFIER_COVARIATE_DIM EFFECT_MODIFIER_SIGNAL_DIM
export EFFECT_MODIFIER_SIGNAL_STRENGTH EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH
export EFFECT_MODIFIER_CONFOUNDING_STRENGTH EFFECT_MODIFIER_INTERCEPT
export EFFECT_MODIFIER_TREATMENT_PROB EFFECT_MODIFIER_PROPENSITY_CLIP
export EFFECT_MODIFIER_SITE_MIN_SAMPLES EFFECT_MODIFIER_SITE_MAX_SAMPLES
export EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN
export EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH
export PREDICTION_INTERVENTION_REPEATS PREDICTION_INTERVENTION_BATCH_SIZE
export OMP_NUM_THREADS MKL_NUM_THREADS

launched=0
for worker_id in "${!gpu_array[@]}"; do
    gpu_id="${gpu_array[${worker_id}]}"
    session="${SESSION_PREFIX}_w${worker_id}"
    if screen -ls 2>/dev/null | grep -Eq "[0-9]+[.]${session}[[:space:]]"; then
        echo "already_running session=${session}"
        continue
    fi
    screen -dmS "${session}" \
        "${SCRIPT_DIR}/run_synthetic_effect_modifier_local_slot.sh" \
        "${worker_id}" "${worker_count}" "${gpu_id}"
    echo "launched session=${session} worker=${worker_id} gpu=${gpu_id}"
    launched=$((launched + 1))
done

sleep 1
echo "effect_modifier_variant=${EFFECT_MODIFIER_VARIANT}"
echo "run_provenance_sha256=${RUN_PROVENANCE_SHA256}"
echo "run_config_sha256=${RUN_CONFIG_SHA256}"
echo "variant_config_sha256=${VARIANT_CONFIG_SHA256}"
echo "git_commit=${GIT_COMMIT} git_dirty=${GIT_DIRTY}"
echo "launched_count=${launched}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_dir=${LOG_DIR}"
echo "status_command=EFFECT_MODIFIER_VARIANT=${EFFECT_MODIFIER_VARIANT} RUN_ID=${RUN_ID} bash scripts/status_synthetic_effect_modifier_local.sh"
screen -ls 2>/dev/null | grep "${SESSION_PREFIX}" || true
