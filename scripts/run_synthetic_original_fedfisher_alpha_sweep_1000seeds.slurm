#!/usr/bin/env bash
#SBATCH --job-name=fedfisher_alpha_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --array=0-9
#SBATCH --output=synthetic_binary_experiment/logs/original_fedfisher_alpha_sweep_1000seeds/%x_%A_%a.out
#SBATCH --error=synthetic_binary_experiment/logs/original_fedfisher_alpha_sweep_1000seeds/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

LOG_DIR="synthetic_binary_experiment/logs/original_fedfisher_alpha_sweep_1000seeds"
OUTPUT_BASE="${OUTPUT_BASE:-synthetic_binary_experiment/outputs/original_fedfisher_alpha_sweep_1000seeds}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_BASE}"

PYTHON_BIN="${PYTHON_BIN:-./.conda/bin/python}"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed_offset="${SEED_OFFSET:-0}"
seeds_per_task="${SEEDS_PER_TASK:-100}"

# New non-IID levels only. Existing baselines are:
# - iid from original_fedfisher_1000seeds
# - noniid alpha=0.1 from original_fedfisher_1000seeds
ALPHAS=(${SYNTHETIC_ALPHAS:-2.0 1.0 0.5 0.02})
seed_start=$((seed_offset + task_id * seeds_per_task))
seed_end=$((seed_start + seeds_per_task - 1))

MODELS=(SyntheticMLP SyntheticMLPDeep)
ALGS=(fedavg fedfisher_diag fedfisher_kfac)

echo "job_id=${SLURM_JOB_ID:-local}"
echo "task_id=${task_id}"
echo "alphas=${ALPHAS[*]}"
echo "seed_start=${seed_start}"
echo "seed_end=${seed_end}"
echo "seeds_per_task=${seeds_per_task}"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "output_base=${OUTPUT_BASE}"
echo "started_at=$(date)"

"${PYTHON_BIN}" -c "import torch, nngeometry; print('env_ok python torch=%s cuda=%s' % (torch.__version__, torch.cuda.is_available())); assert torch.cuda.is_available(), 'CUDA is not available on this allocated node'"

for ((seed = seed_start; seed <= seed_end; seed++)); do
  echo "running seed=${seed} started_at=$(date)"
  for alpha in "${ALPHAS[@]}"; do
    alpha_tag="${alpha//./p}"
    output_dir="${OUTPUT_BASE}/alpha${alpha_tag}"
    mkdir -p "${output_dir}"
    for model in "${MODELS[@]}"; do
      echo "running seed=${seed} alpha=${alpha} split=noniid model=${model} started_at=$(date)"
      "${PYTHON_BIN}" main.py \
        --dataset SyntheticBinary \
        --model "${model}" \
        --seed "${seed}" \
        --alpha "${alpha}" \
        --num_clients "${SYNTHETIC_NUM_CLIENTS:-5}" \
        --num_rounds 1 \
        --local_epochs "${SYNTHETIC_LOCAL_EPOCHS:-30}" \
        --output_dir "${output_dir}" \
        --synthetic_split noniid \
        --synthetic_num_train "${SYNTHETIC_NUM_TRAIN:-10000}" \
        --synthetic_num_test "${SYNTHETIC_NUM_TEST:-10000}" \
        --synthetic_dim "${SYNTHETIC_DIM:-100}" \
        --synthetic_signal_dim "${SYNTHETIC_SIGNAL_DIM:-10}" \
        --synthetic_signal_strength "${SYNTHETIC_SIGNAL_STRENGTH:-0.7}" \
        --synthetic_noise_std "${SYNTHETIC_NOISE_STD:-1.0}" \
        --algs_to_run "${ALGS[@]}"
      echo "finished seed=${seed} alpha=${alpha} split=noniid model=${model} finished_at=$(date)"
    done
  done
  echo "finished seed=${seed} finished_at=$(date)"
done

echo "finished_at=$(date)"
