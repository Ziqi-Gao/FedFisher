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
        site_gaussian_local_permutation)
            [[ "${RUN_ID}" == site_gaussian_local_permutation_* ]] \
                || fail "site_gaussian_local_permutation RUN_ID must start with site_gaussian_local_permutation_"
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

validate_metadata_field() {
    local path="$1"
    local key="$2"
    local expected="$3"
    local actual

    actual="$(metadata_value "${path}" "${key}")"         || fail "Metadata has missing or duplicate key ${key}: ${path}"
    [[ "${actual}" == "${expected}" ]]         || fail "Metadata mismatch for ${key}: expected ${expected}, found ${actual}"
}

validate_run_manifest() {
    local manifest="$1"

    [[ -f "${manifest}" && ! -L "${manifest}" ]]         || fail "Run manifest must be a regular non-symlink file: ${manifest}"
    validate_metadata_field "${manifest}" "format_version" "2"
    validate_metadata_field "${manifest}" "run_id" "${RUN_ID}"
    validate_metadata_field         "${manifest}"         "effect_modifier_variant"         "${EFFECT_MODIFIER_VARIANT}"
    validate_metadata_field "${manifest}" "git_commit" "${GIT_COMMIT}"
    validate_metadata_field "${manifest}" "git_dirty" "${GIT_DIRTY}"
    validate_metadata_field         "${manifest}"         "git_status_sha256"         "${GIT_STATUS_SHA256}"
    validate_metadata_field         "${manifest}"         "source_tree_sha256"         "${SOURCE_TREE_SHA256}"
    validate_metadata_field         "${manifest}"         "provenance_sha256"         "${RUN_PROVENANCE_SHA256}"
    validate_metadata_field         "${manifest}"         "run_config_sha256"         "${RUN_CONFIG_SHA256}"
    validate_metadata_field         "${manifest}"         "variant_config_sha256"         "${VARIANT_CONFIG_SHA256}"
    validate_metadata_field         "${manifest}"         "effect_modifier_site_gaussian_shift"         "${SITE_GAUSSIAN_SHIFT_ENABLED}"
    validate_metadata_field         "${manifest}"         "prediction_intervention_scope"         "${PREDICTION_INTERVENTION_SCOPE}"
}

validate_current_source_provenance() {
    local current_commit
    local current_status
    local current_dirty
    local current_status_sha256
    local current_source_tree_sha256
    local current_provenance_sha256

    current_commit="$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)"
    current_status="$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)"
    if [[ -n "${current_status}" ]]; then
        current_dirty=1
    else
        current_dirty=0
    fi
    current_status_sha256="$(
        printf '%s\n' "${current_status}" | sha256sum | awk '{print $1}'
    )"
    current_source_tree_sha256="$(compute_source_tree_sha256)"
    current_provenance_sha256="$(
        printf '%s\n' \
            "git_commit=${current_commit}" \
            "git_dirty=${current_dirty}" \
            "git_status_sha256=${current_status_sha256}" \
            "source_tree_sha256=${current_source_tree_sha256}" \
            | sha256sum \
            | awk '{print $1}'
    )"
    [[ "${current_provenance_sha256}" == "${RUN_PROVENANCE_SHA256}" ]] \
        || fail "Source tree changed after launch; refusing to mix code in RUN_ID=${RUN_ID}"
}

validate_completion_marker() {
    local marker="$1"
    local expected_seed="$2"

    [[ -f "${marker}" && ! -L "${marker}" ]]         || fail "Completion marker must be a regular non-symlink file: ${marker}"
    validate_metadata_field "${marker}" "format_version" "2"
    validate_metadata_field "${marker}" "seed" "${expected_seed}"
    validate_metadata_field "${marker}" "run_id" "${RUN_ID}"
    validate_metadata_field         "${marker}"         "effect_modifier_variant"         "${EFFECT_MODIFIER_VARIANT}"
    validate_metadata_field         "${marker}"         "provenance_sha256"         "${RUN_PROVENANCE_SHA256}"
    validate_metadata_field         "${marker}"         "run_config_sha256"         "${RUN_CONFIG_SHA256}"
    validate_metadata_field         "${marker}"         "variant_config_sha256"         "${VARIANT_CONFIG_SHA256}"
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

for required_command in realpath git tar sha256sum awk find ln; do
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
        variant_args=()
        ;;
    site_gaussian_shift)
        default_run_id="site_gaussian_shift_300seeds"
        SITE_GAUSSIAN_SHIFT_ENABLED=1
        PREDICTION_INTERVENTION_SCOPE=global
        variant_args=(
            --effect_modifier_site_gaussian_shift
            --effect_modifier_site_gaussian_mean_span
            "${EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN:-0.75}"
            --effect_modifier_site_gaussian_std_low
            "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW:-0.75}"
            --effect_modifier_site_gaussian_std_high
            "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH:-1.25}"
        )
        ;;
    local_permutation)
        default_run_id="local_permutation_300seeds"
        SITE_GAUSSIAN_SHIFT_ENABLED=0
        PREDICTION_INTERVENTION_SCOPE=site
        variant_args=(--prediction_intervention_scope site)
        ;;
    site_gaussian_local_permutation)
        default_run_id="site_gaussian_local_permutation_300seeds"
        SITE_GAUSSIAN_SHIFT_ENABLED=1
        PREDICTION_INTERVENTION_SCOPE=site
        variant_args=(
            --effect_modifier_combined_ablation
            --effect_modifier_site_gaussian_shift
            --effect_modifier_site_gaussian_mean_span
            "${EFFECT_MODIFIER_SITE_GAUSSIAN_MEAN_SPAN:-0.75}"
            --effect_modifier_site_gaussian_std_low
            "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_LOW:-0.75}"
            --effect_modifier_site_gaussian_std_high
            "${EFFECT_MODIFIER_SITE_GAUSSIAN_STD_HIGH:-1.25}"
            --prediction_intervention_scope
            site
        )
        ;;
    *)
        fail "EFFECT_MODIFIER_VARIANT must be control, site_gaussian_shift, local_permutation, or site_gaussian_local_permutation"
        ;;
esac

RUN_ID="${RUN_ID:-${default_run_id}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/effect_modifier_experiment/outputs/${RUN_ID}}"
DONE_DIR="${OUTPUT_DIR}/completed_seeds"
RUNTIME_DIR="${SCRATCH_ROOT}/effect_modifier_experiment/runtime/${RUN_ID}/worker_${worker_id}"
PYTHON_BIN="${PYTHON_BIN:-${SCRATCH_ROOT}/envs/fedfisher-py39/bin/python}"
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

[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]     || fail "RUN_ID may only contain letters, digits, dot, underscore, and hyphen"
validate_run_id_namespace
for value in "${SEED_START}" "${SEED_END}"; do
    [[ "${value}" =~ ^[0-9]+$ ]]         || fail "SEED_START and SEED_END must be non-negative integers"
done
(( SEED_END >= SEED_START ))     || fail "SEED_END must be greater than or equal to SEED_START"

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

REPO_ROOT="$(canonical_exact "REPO_ROOT" "${REPO_ROOT}" "${FEDFISHER_PROJECT_ROOT}")"
DATA_ROOT="$(canonical_exact "DATA_ROOT" "${DATA_ROOT}" "${FEDFISHER_DATA_ROOT}")"
SCRATCH_ROOT="$(canonical_exact "SCRATCH_ROOT" "${SCRATCH_ROOT}" "${FEDFISHER_SCRATCH_ROOT}")"
OUTPUT_DIR="$(canonical_bound_path "OUTPUT_DIR" "${OUTPUT_DIR}" "${FEDFISHER_DATA_ROOT}" "effect_modifier_experiment/outputs/${RUN_ID}")"
DONE_DIR="$(canonical_bound_path "DONE_DIR" "${DONE_DIR}" "${FEDFISHER_DATA_ROOT}" "effect_modifier_experiment/outputs/${RUN_ID}/completed_seeds")"
RUNTIME_DIR="$(canonical_bound_path "RUNTIME_DIR" "${RUNTIME_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "effect_modifier_experiment/runtime/${RUN_ID}/worker_${worker_id}")"
RUN_MANIFEST="${OUTPUT_DIR}/run_manifest.env"

[[ -d "${OUTPUT_DIR}" ]]     || fail "OUTPUT_DIR must be initialized by the existing launcher: ${OUTPUT_DIR}"
[[ -f "${RUN_MANIFEST}" && ! -L "${RUN_MANIFEST}" ]]     || fail "Missing safe run manifest; use the existing launcher: ${RUN_MANIFEST}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi

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
validate_run_manifest "${RUN_MANIFEST}"
unsafe_output_symlink="$(find "${OUTPUT_DIR}" -type l -print -quit)"
[[ -z "${unsafe_output_symlink}" ]] \
    || fail "Output tree may not contain symlinks: ${unsafe_output_symlink}"

mkdir -p "${DONE_DIR}" "${RUNTIME_DIR}"
DONE_DIR="$(canonical_bound_path "DONE_DIR" "${DONE_DIR}" "${FEDFISHER_DATA_ROOT}" "effect_modifier_experiment/outputs/${RUN_ID}/completed_seeds")"
RUNTIME_DIR="$(canonical_bound_path "RUNTIME_DIR" "${RUNTIME_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "effect_modifier_experiment/runtime/${RUN_ID}/worker_${worker_id}")"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export OMP_NUM_THREADS
export MKL_NUM_THREADS
runtime_relative="effect_modifier_experiment/runtime/${RUN_ID}/worker_${worker_id}"
TMPDIR="$(canonical_bound_path "TMPDIR" "${RUNTIME_DIR}/tmp" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/tmp")"
PYTHONPYCACHEPREFIX="$(canonical_bound_path "PYTHONPYCACHEPREFIX" "${RUNTIME_DIR}/python_cache" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/python_cache")"
XDG_CACHE_HOME="$(canonical_bound_path "XDG_CACHE_HOME" "${RUNTIME_DIR}/xdg_cache" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/xdg_cache")"
CUDA_CACHE_PATH="$(canonical_bound_path "CUDA_CACHE_PATH" "${RUNTIME_DIR}/cuda_cache" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/cuda_cache")"
TRITON_CACHE_DIR="$(canonical_bound_path "TRITON_CACHE_DIR" "${RUNTIME_DIR}/triton_cache" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/triton_cache")"
TORCH_HOME="$(canonical_bound_path "TORCH_HOME" "${RUNTIME_DIR}/torch_cache" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/torch_cache")"
export TMPDIR PYTHONPYCACHEPREFIX XDG_CACHE_HOME CUDA_CACHE_PATH TRITON_CACHE_DIR TORCH_HOME
mkdir -p \
    "${TMPDIR}" \
    "${PYTHONPYCACHEPREFIX}" \
    "${XDG_CACHE_HOME}" \
    "${CUDA_CACHE_PATH}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_HOME}"
canonical_bound_path "TMPDIR" "${TMPDIR}" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/tmp" >/dev/null
canonical_bound_path "PYTHONPYCACHEPREFIX" "${PYTHONPYCACHEPREFIX}" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/python_cache" >/dev/null
canonical_bound_path "XDG_CACHE_HOME" "${XDG_CACHE_HOME}" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/xdg_cache" >/dev/null
canonical_bound_path "CUDA_CACHE_PATH" "${CUDA_CACHE_PATH}" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/cuda_cache" >/dev/null
canonical_bound_path "TRITON_CACHE_DIR" "${TRITON_CACHE_DIR}" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/triton_cache" >/dev/null
canonical_bound_path "TORCH_HOME" "${TORCH_HOME}" "${FEDFISHER_SCRATCH_ROOT}" "${runtime_relative}/torch_cache" >/dev/null

cd "${REPO_ROOT}"
"${PYTHON_BIN}" - <<'PY_CHECK'
import nngeometry
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "Worker must see exactly one GPU"
assert "sm_120" in torch.cuda.get_arch_list(), "PyTorch lacks sm_120 support"
x = torch.ones(1, device="cuda")
assert (x + x).cpu().item() == 2.0
print("worker_environment_ok", torch.__version__, torch.cuda.get_device_name(0), nngeometry.__file__)
PY_CHECK

models=(SyntheticMLP SyntheticMLPDeep)
splits=(iid noniid)
algs=(fedavg fedfisher_diag fedfisher_kfac)
first_seed=$((SEED_START + worker_id))

echo "effect_modifier_variant=${EFFECT_MODIFIER_VARIANT}"
echo "run_id=${RUN_ID}"
echo "run_provenance_sha256=${RUN_PROVENANCE_SHA256}"
echo "run_config_sha256=${RUN_CONFIG_SHA256}"
echo "variant_config_sha256=${VARIANT_CONFIG_SHA256}"
echo "git_commit=${GIT_COMMIT} git_dirty=${GIT_DIRTY}"
echo "worker_id=${worker_id} worker_count=${worker_count} physical_gpu=${gpu_id}"
echo "seed_sequence=${first_seed}..${SEED_END} step=${worker_count}"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "output_dir=${OUTPUT_DIR}"
echo "started_at=$(date -Is)"

for ((seed = first_seed; seed <= SEED_END; seed += worker_count)); do
    marker="${DONE_DIR}/seed_${seed}.done"
    if [[ -e "${marker}" || -L "${marker}" ]]; then
        validate_completion_marker "${marker}" "${seed}"
        echo "skipping validated completed seed=${seed}"
        continue
    fi

    echo "running seed=${seed} started_at=$(date -Is)"
    for model in "${models[@]}"; do
        for split in "${splits[@]}"; do
            echo "running seed=${seed} split=${split} model=${model} started_at=$(date -Is)"
            validate_current_source_provenance
            "${PYTHON_BIN}" main.py \
                --dataset SyntheticEffectModifier \
                --model "${model}" \
                --seed "${seed}" \
                --alpha "${SYNTHETIC_ALPHA}" \
                --num_clients "${EFFECT_MODIFIER_NUM_SITES}" \
                --num_rounds 1 \
                --local_epochs "${SYNTHETIC_LOCAL_EPOCHS}" \
                --output_dir "${OUTPUT_DIR}" \
                --synthetic_split "${split}" \
                --synthetic_num_test "${SYNTHETIC_NUM_TEST}" \
                --effect_modifier_covariate_dim "${EFFECT_MODIFIER_COVARIATE_DIM}" \
                --effect_modifier_signal_dim "${EFFECT_MODIFIER_SIGNAL_DIM}" \
                --effect_modifier_signal_strength "${EFFECT_MODIFIER_SIGNAL_STRENGTH}" \
                --effect_modifier_main_effect_strength "${EFFECT_MODIFIER_MAIN_EFFECT_STRENGTH}" \
                --effect_modifier_confounding_strength "${EFFECT_MODIFIER_CONFOUNDING_STRENGTH}" \
                --effect_modifier_intercept "${EFFECT_MODIFIER_INTERCEPT}" \
                --effect_modifier_treatment_prob "${EFFECT_MODIFIER_TREATMENT_PROB}" \
                --effect_modifier_propensity_clip "${EFFECT_MODIFIER_PROPENSITY_CLIP}" \
                --effect_modifier_site_min_samples "${EFFECT_MODIFIER_SITE_MIN_SAMPLES}" \
                --effect_modifier_site_max_samples "${EFFECT_MODIFIER_SITE_MAX_SAMPLES}" \
                --algs_to_run "${algs[@]}" \
                --prediction_intervention \
                --prediction_intervention_modes permute zero \
                --prediction_intervention_repeats "${PREDICTION_INTERVENTION_REPEATS}" \
                --prediction_intervention_batch_size "${PREDICTION_INTERVENTION_BATCH_SIZE}" \
                "${variant_args[@]}"
            echo "finished seed=${seed} split=${split} model=${model} finished_at=$(date -Is)"
        done
    done

    marker_tmp="${marker}.tmp.$$"
    [[ ! -e "${marker_tmp}" && ! -L "${marker_tmp}" ]]         || fail "Temporary completion marker already exists: ${marker_tmp}"
    (
        umask 077
        printf '%s\n' \
            "format_version=2" \
            "seed=${seed}" \
            "completed_at=$(date -Is)" \
            "worker_id=${worker_id}" \
            "gpu_id=${gpu_id}" \
            "run_id=${RUN_ID}" \
            "effect_modifier_variant=${EFFECT_MODIFIER_VARIANT}" \
            "provenance_sha256=${RUN_PROVENANCE_SHA256}" \
            "run_config_sha256=${RUN_CONFIG_SHA256}" \
            "variant_config_sha256=${VARIANT_CONFIG_SHA256}" > "${marker_tmp}"
    )
    if ! ln -- "${marker_tmp}" "${marker}" 2>/dev/null; then
        rm -- "${marker_tmp}"
        validate_completion_marker "${marker}" "${seed}"
    else
        rm -- "${marker_tmp}"
    fi
    echo "finished seed=${seed} finished_at=$(date -Is)"
done

echo "worker_finished_at=$(date -Is)"
