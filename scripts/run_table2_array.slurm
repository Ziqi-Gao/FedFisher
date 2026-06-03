#!/usr/bin/env bash
#SBATCH --job-name=fedfisher_t2
#SBATCH --account=p32737
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-59
#SBATCH --output=logs/table2/%x_%A_%a.out
#SBATCH --error=logs/table2/%x_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/table2 results/table2

PYTHON_BIN="${PYTHON_BIN:-./.conda/bin/python}"

DATASETS=(FashionMNIST SVHN CIFAR10 CINIC10)
MODELS=(LeNet CNN CNN CNN)
ALPHAS=(0.2 0.1 0.05)
SEEDS=(0 1 2 3 4)
ALGS=(fedavg otfusion pfnm regmean dense fisher_merge fedfisher_diag fedfisher_kfac)

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
num_alphas="${#ALPHAS[@]}"
num_seeds="${#SEEDS[@]}"

dataset_idx=$((task_id / (num_alphas * num_seeds)))
rem=$((task_id % (num_alphas * num_seeds)))
alpha_idx=$((rem / num_seeds))
seed_idx=$((rem % num_seeds))

dataset="${DATASETS[$dataset_idx]}"
model="${MODELS[$dataset_idx]}"
alpha="${ALPHAS[$alpha_idx]}"
seed="${SEEDS[$seed_idx]}"

echo "dataset=${dataset} model=${model} alpha=${alpha} seed=${seed} clients=5 rounds=1"
echo "python=${PYTHON_BIN}"

"${PYTHON_BIN}" main.py \
  --dataset "${dataset}" \
  --model "${model}" \
  --seed "${seed}" \
  --alpha "${alpha}" \
  --num_clients 5 \
  --num_rounds 1 \
  --local_epochs 30 \
  --output_dir results/table2 \
  --algs_to_run "${ALGS[@]}"
