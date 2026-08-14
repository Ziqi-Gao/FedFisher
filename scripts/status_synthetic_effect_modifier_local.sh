#!/usr/bin/env bash
set -euo pipefail

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
    [[ "${resolved_candidate}" == "${resolved_expected}" ]] \
        || fail "${label} must resolve to ${expected}: ${resolved_candidate}"
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
            [[ "${RUN_ID}" == local_permutation_* ]] \
                || fail "local_permutation RUN_ID must start with local_permutation_"
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

field_matches() {
    local path="$1"
    local key="$2"
    local expected="$3"
    local actual

    actual="$(metadata_value "${path}" "${key}")" || return 1
    [[ "${actual}" == "${expected}" ]]
}

for required_command in realpath sha256sum; do
    command -v "${required_command}" >/dev/null 2>&1 \
        || fail "Required command is unavailable: ${required_command}"
done

SCRATCH_ROOT="${SCRATCH_ROOT:-${FEDFISHER_SCRATCH_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${FEDFISHER_DATA_ROOT}}"
EFFECT_MODIFIER_VARIANT="${EFFECT_MODIFIER_VARIANT:-control}"

case "${EFFECT_MODIFIER_VARIANT}" in
    control)
        default_run_id="prediction_intervention_ablation_control_300seeds"
        expected_site_gaussian_shift=0
        expected_intervention_scope=global
        ;;
    site_gaussian_shift)
        default_run_id="site_gaussian_shift_300seeds"
        expected_site_gaussian_shift=1
        expected_intervention_scope=global
        ;;
    local_permutation)
        default_run_id="local_permutation_300seeds"
        expected_site_gaussian_shift=0
        expected_intervention_scope=site
        ;;
    *)
        fail "EFFECT_MODIFIER_VARIANT must be control, site_gaussian_shift, or local_permutation"
        ;;
esac

RUN_ID="${RUN_ID:-${default_run_id}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] \
    || fail "RUN_ID may only contain letters, digits, dot, underscore, and hyphen"
validate_run_id_namespace

DATA_ROOT="$(canonical_exact "DATA_ROOT" "${DATA_ROOT}" "${FEDFISHER_DATA_ROOT}")"
SCRATCH_ROOT="$(canonical_exact "SCRATCH_ROOT" "${SCRATCH_ROOT}" "${FEDFISHER_SCRATCH_ROOT}")"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/effect_modifier_experiment/outputs/${RUN_ID}}"
OUTPUT_DIR="$(canonical_bound_path "OUTPUT_DIR" "${OUTPUT_DIR}" "${FEDFISHER_DATA_ROOT}" "effect_modifier_experiment/outputs/${RUN_ID}")"
LOG_DIR="$(canonical_bound_path \
    "LOG_DIR" \
    "${SCRATCH_ROOT}/effect_modifier_experiment/logs/${RUN_ID}" \
    "${FEDFISHER_SCRATCH_ROOT}" \
    "effect_modifier_experiment/logs/${RUN_ID}")"
SESSION_PREFIX="fedfisher_em_${RUN_ID}"
RUN_MANIFEST="${OUTPUT_DIR}/run_manifest.env"

printf 'Effect-modifier variant: %s\n' "${EFFECT_MODIFIER_VARIANT}"
printf 'RUN_ID: %s\n' "${RUN_ID}"
printf 'Active sessions:\n'
screen -ls 2>/dev/null | grep "${SESSION_PREFIX}" || printf '  none\n'

manifest_valid=0
manifest_status="missing"
manifest_provenance=""
manifest_config=""
manifest_variant_config=""
configured_seed_start=""
configured_seed_end=""
configured_seed_total="unknown"
if [[ -f "${RUN_MANIFEST}" && ! -L "${RUN_MANIFEST}" ]]; then
    if field_matches "${RUN_MANIFEST}" "format_version" "2" \
        && field_matches "${RUN_MANIFEST}" "run_id" "${RUN_ID}" \
        && field_matches \
            "${RUN_MANIFEST}" \
            "effect_modifier_variant" \
            "${EFFECT_MODIFIER_VARIANT}"; then
        manifest_provenance="$(metadata_value "${RUN_MANIFEST}" "provenance_sha256")" \
            || fail "Run manifest is missing provenance_sha256: ${RUN_MANIFEST}"
        manifest_config="$(metadata_value "${RUN_MANIFEST}" "run_config_sha256")" \
            || fail "Run manifest is missing run_config_sha256: ${RUN_MANIFEST}"
        manifest_variant_config="$(metadata_value "${RUN_MANIFEST}" "variant_config_sha256")" \
            || fail "Run manifest is missing variant_config_sha256: ${RUN_MANIFEST}"
        configured_seed_start="$(metadata_value "${RUN_MANIFEST}" "seed_start")" \
            || fail "Run manifest is missing seed_start: ${RUN_MANIFEST}"
        configured_seed_end="$(metadata_value "${RUN_MANIFEST}" "seed_end")" \
            || fail "Run manifest is missing seed_end: ${RUN_MANIFEST}"
        [[ "${configured_seed_start}" =~ ^[0-9]+$ \
            && "${configured_seed_end}" =~ ^[0-9]+$ ]] \
            || fail "Run manifest seed range must contain non-negative integers"
        (( configured_seed_end >= configured_seed_start )) \
            || fail "Run manifest seed_end must be >= seed_start"
        configured_seed_total=$((configured_seed_end - configured_seed_start + 1))
        field_matches "${RUN_MANIFEST}" "effect_modifier_site_gaussian_shift" "${expected_site_gaussian_shift}" \
            || fail "Run manifest has the wrong site-Gaussian setting: ${RUN_MANIFEST}"
        field_matches "${RUN_MANIFEST}" "prediction_intervention_scope" "${expected_intervention_scope}" \
            || fail "Run manifest has the wrong intervention scope: ${RUN_MANIFEST}"
        expected_variant_config="$(
            printf '%s\n' \
                "effect_modifier_site_gaussian_shift=${expected_site_gaussian_shift}" \
                "prediction_intervention_scope=${expected_intervention_scope}" \
                | sha256sum | awk '{print $1}'
        )"
        [[ "${manifest_variant_config}" == "${expected_variant_config}" ]] \
            || fail "Run manifest variant_config_sha256 is inconsistent: ${RUN_MANIFEST}"
        manifest_valid=1
        manifest_status="validated"
    else
        fail "Run manifest does not match RUN_ID/variant: ${RUN_MANIFEST}"
    fi
elif [[ -e "${RUN_MANIFEST}" || -L "${RUN_MANIFEST}" ]]; then
    fail "Run manifest is not a regular non-symlink file: ${RUN_MANIFEST}"
elif [[ -d "${OUTPUT_DIR}" ]] \
    && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    manifest_status="legacy_or_unsafe_nonempty_output"
fi

completed=0
invalid_markers=0
legacy_markers=0
marker_dir="$(canonical_bound_path \
    "completed marker directory" \
    "${OUTPUT_DIR}/completed_seeds" \
    "${FEDFISHER_DATA_ROOT}" \
    "effect_modifier_experiment/outputs/${RUN_ID}/completed_seeds")"
if [[ -d "${marker_dir}" ]]; then
    while IFS= read -r -d '' marker; do
        marker_name="${marker##*/}"
        expected_seed="${marker_name#seed_}"
        expected_seed="${expected_seed%.done}"
        if (( manifest_valid == 1 )) \
            && [[ -f "${marker}" && ! -L "${marker}" ]] \
            && [[ "${expected_seed}" =~ ^[0-9]+$ ]] \
            && (( expected_seed >= configured_seed_start && expected_seed <= configured_seed_end )) \
            && field_matches "${marker}" "format_version" "2" \
            && field_matches "${marker}" "seed" "${expected_seed}" \
            && field_matches "${marker}" "run_id" "${RUN_ID}" \
            && field_matches \
                "${marker}" \
                "effect_modifier_variant" \
                "${EFFECT_MODIFIER_VARIANT}" \
            && field_matches \
                "${marker}" \
                "provenance_sha256" \
                "${manifest_provenance}" \
            && field_matches \
                "${marker}" \
                "run_config_sha256" \
                "${manifest_config}" \
            && field_matches \
                "${marker}" \
                "variant_config_sha256" \
                "${manifest_variant_config}"; then
            completed=$((completed + 1))
        elif (( manifest_valid == 0 )); then
            legacy_markers=$((legacy_markers + 1))
        else
            invalid_markers=$((invalid_markers + 1))
        fi
    done < <(find "${marker_dir}" -maxdepth 1 -name 'seed_*.done' -print0)
fi

printf '\nManifest status: %s\n' "${manifest_status}"
if (( manifest_valid == 1 )); then
    printf 'Provenance SHA256: %s\n' "${manifest_provenance}"
    printf 'Run-config SHA256: %s\n' "${manifest_config}"
    printf 'Variant-config SHA256: %s\n' "${manifest_variant_config}"
    printf 'Configured seed range: %s..%s\n' "${configured_seed_start}" "${configured_seed_end}"
fi
printf 'Validated completed seeds: %s / %s\n' "${completed}" "${configured_seed_total}"
printf 'Invalid/mismatched markers: %s\n' "${invalid_markers}"
printf 'Legacy/unvalidated markers: %s\n' "${legacy_markers}"
printf 'Durable output: %s\n' "${OUTPUT_DIR}"
printf 'Scratch logs: %s\n' "${LOG_DIR}"

printf '\nGPU processes:\n'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader || true

printf '\nWorker log tails:\n'
shopt -s nullglob
logs=("${LOG_DIR}"/worker_*.log)
if (( ${#logs[@]} == 0 )); then
    printf '  no logs yet\n'
else
    for log in "${logs[@]}"; do
        if [[ ! -f "${log}" || -L "${log}" ]]; then
            printf '  skipping unsafe log path: %s\n' "${log}"
            continue
        fi
        printf '\n==> %s <==\n' "${log}"
        tail -n 8 "${log}"
    done
fi
