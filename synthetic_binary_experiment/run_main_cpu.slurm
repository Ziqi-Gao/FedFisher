#!/usr/bin/env bash
#SBATCH --job-name=fedfisher_synth
#SBATCH --account=p32737
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=synthetic_binary_experiment/logs/%x_%j.out
#SBATCH --error=synthetic_binary_experiment/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gpfs/projects/p32737/del6500_home/FedFisher}"
cd "${REPO_ROOT}"

mkdir -p synthetic_binary_experiment/logs synthetic_binary_experiment/outputs/main_cpu

PYTHON_BIN="${PYTHON_BIN:-./.conda/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-synthetic_binary_experiment/outputs/main_cpu}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "repo=${REPO_ROOT}"
echo "python=${PYTHON_BIN}"
echo "output_dir=${OUTPUT_DIR}"
echo "started_at=$(date)"

"${PYTHON_BIN}" synthetic_binary_experiment/run_experiment.py \
  --output-dir "${OUTPUT_DIR}" \
  --num-train 10000 \
  --num-test 10000 \
  --dim 100 \
  --signal-dim 10 \
  --signal-strength 0.7 \
  --noise-std 1.0 \
  --num-clients 5 \
  --dirichlet-alpha 0.1 \
  --local-epochs 30 \
  --optimizer sgd \
  --weight-decay 0.0 \
  --pool-epochs 80 \
  --pool-lr 1e-3 \
  --pool-weight-decay 1e-2 \
  --pool-optimizer adam \
  --pool-val-fraction 0.2 \
  --pool-patience 8 \
  --batch-size 128 \
  --fisher-batch-size 128 \
  --fisher-damping 1e-6 \
  --fisher-server-steps 2000 \
  --fisher-server-lr 1e-2 \
  --fisher-server-eval-every 100 \
  --fisher-val-size 500 \
  --seeds 0 1 2 3 4 \
  --model-types lr mlp \
  --splits iid noniid \
  --device cpu

"${PYTHON_BIN}" synthetic_binary_experiment/summarize_results.py \
  --input "${OUTPUT_DIR}/results.csv" \
  --output "${OUTPUT_DIR}/summary.csv"

"${PYTHON_BIN}" synthetic_binary_experiment/plot_results.py \
  --input "${OUTPUT_DIR}/results.csv" \
  --output-dir "${OUTPUT_DIR}/figures"

echo "finished_at=$(date)"
echo "results=${OUTPUT_DIR}/results.csv"
echo "summary=${OUTPUT_DIR}/summary.csv"
echo "figures=${OUTPUT_DIR}/figures"
