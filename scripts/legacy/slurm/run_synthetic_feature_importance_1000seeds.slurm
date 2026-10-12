#!/usr/bin/env bash
#SBATCH --job-name=fedfisher_fi_1000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --array=0-49
#SBATCH --chdir=/gpfs/projects/p32737/del6500_home/FedFisher
#SBATCH --output=/gpfs/projects/p32737/del6500_home/FedFisher/synthetic_binary_experiment/logs/feature_importance_1000seeds/%x_%A_%a.out
#SBATCH --error=/gpfs/projects/p32737/del6500_home/FedFisher/synthetic_binary_experiment/logs/feature_importance_1000seeds/%x_%A_%a.err

set -euo pipefail

# Slurm executes a copied script from its spool directory, so BASH_SOURCE[0]
# cannot be used to recover the repository path inside the allocation.
REPO_ROOT="${REPO_ROOT:-/gpfs/projects/p32737/del6500_home/FedFisher}"
cd "${REPO_ROOT}"

LOG_DIR="synthetic_binary_experiment/logs/feature_importance_1000seeds"
OUTPUT_DIR="${OUTPUT_DIR:-synthetic_binary_experiment/outputs/feature_importance_1000seeds}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-./.conda/bin/python}"

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
seed_offset="${SEED_OFFSET:-0}"
seeds_per_task="${SEEDS_PER_TASK:-20}"
seed_start=$((seed_offset + task_id * seeds_per_task))
seed_end=$((seed_start + seeds_per_task - 1))

MODELS=(SyntheticMLP SyntheticMLPDeep)
SPLITS=(iid noniid)
ALGS=(fedavg fedfisher_diag fedfisher_kfac)

echo "job_id=${SLURM_JOB_ID:-local}"
echo "task_id=${task_id}"
echo "seed_start=${seed_start}"
echo "seed_end=${seed_end}"
echo "seeds_per_task=${seeds_per_task}"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "output_dir=${OUTPUT_DIR}"
echo "started_at=$(date)"

"${PYTHON_BIN}" -c "import torch, nngeometry; print('env_ok python torch=%s cuda=%s' % (torch.__version__, torch.cuda.is_available())); assert torch.cuda.is_available(), 'CUDA is not available on this allocated node'"

for ((seed = seed_start; seed <= seed_end; seed++)); do
  echo "running seed=${seed} started_at=$(date)"
  for model in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
      echo "running seed=${seed} split=${split} model=${model} started_at=$(date)"
      "${PYTHON_BIN}" main.py \
        --dataset SyntheticBinary \
        --model "${model}" \
        --seed "${seed}" \
        --alpha "${SYNTHETIC_ALPHA:-0.1}" \
        --num_clients "${SYNTHETIC_NUM_CLIENTS:-5}" \
        --num_rounds 1 \
        --local_epochs "${SYNTHETIC_LOCAL_EPOCHS:-30}" \
        --output_dir "${OUTPUT_DIR}" \
        --synthetic_split "${split}" \
        --synthetic_num_train "${SYNTHETIC_NUM_TRAIN:-10000}" \
        --synthetic_num_test "${SYNTHETIC_NUM_TEST:-10000}" \
        --synthetic_dim "${SYNTHETIC_DIM:-100}" \
        --synthetic_signal_dim "${SYNTHETIC_SIGNAL_DIM:-10}" \
        --synthetic_signal_strength "${SYNTHETIC_SIGNAL_STRENGTH:-0.7}" \
        --synthetic_noise_std "${SYNTHETIC_NOISE_STD:-1.0}" \
        --algs_to_run "${ALGS[@]}" \
        --feature_importance \
        --feature_importance_repeats "${FEATURE_IMPORTANCE_REPEATS:-5}" \
        --feature_importance_batch_size "${FEATURE_IMPORTANCE_BATCH_SIZE:-1024}"
      echo "finished seed=${seed} split=${split} model=${model} finished_at=$(date)"
    done
  done
  echo "finished seed=${seed} finished_at=$(date)"
done

echo "finished_at=$(date)"
