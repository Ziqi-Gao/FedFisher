#!/usr/bin/env bash
#SBATCH --job-name=fedfisher_synth_orig
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --output=synthetic_binary_experiment/logs/original_fedfisher/%x_%A_%a.out
#SBATCH --error=synthetic_binary_experiment/logs/original_fedfisher/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/projects/p32737/del6500_home/FedFisher}"
cd "${REPO_ROOT}"

mkdir -p synthetic_binary_experiment/logs/original_fedfisher
mkdir -p synthetic_binary_experiment/outputs/original_fedfisher

PYTHON_BIN="${PYTHON_BIN:-./.conda/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-synthetic_binary_experiment/outputs/original_fedfisher}"

SEEDS=(0 1 2 3 4)
SPLITS=(iid noniid)
MODELS=(SyntheticMLP SyntheticMLPDeep)
ALGS=(fedavg fedfisher_diag fedfisher_kfac)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
num_seeds="${#SEEDS[@]}"
num_splits="${#SPLITS[@]}"

model_idx=$((task_id / (num_seeds * num_splits)))
rem=$((task_id % (num_seeds * num_splits)))
split_idx=$((rem / num_seeds))
seed_idx=$((rem % num_seeds))
model="${MODELS[$model_idx]}"
split="${SPLITS[$split_idx]}"
seed="${SEEDS[$seed_idx]}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "task_id=${task_id}"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "output_dir=${OUTPUT_DIR}"
echo "seed=${seed} split=${split} model=${model}"
echo "started_at=$(date)"

"${PYTHON_BIN}" -c "import torch, torchvision, nngeometry; print('env_ok python torch=%s torchvision=%s cuda=%s' % (torch.__version__, torchvision.__version__, torch.cuda.is_available())); assert torch.cuda.is_available(), 'CUDA is not available on this allocated node'"

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
  --algs_to_run "${ALGS[@]}"

echo "finished_at=$(date)"
