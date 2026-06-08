# FedFisher Binary Tabular Pipeline

This repository is adapted from the FedFisher codebase for one-shot federated
learning. The pipeline in this branch is scoped to binary tabular
classification, with a synthetic generator for sanity checks and a CSV loader
for local datasets.

The original image datasets and image models have been removed. The remaining
entry points are:

```text
data.py                                      # SyntheticBinary and LocalBinaryCSV data loaders
models.py                                    # SyntheticMLP and SyntheticMLPDeep
main.py                                      # one-shot FL runner
scripts/run_synthetic_original_fedfisher.slurm
scripts/run_synthetic_original_fedfisher_1000seeds.slurm
scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
scripts/summarize_synthetic_original.py
scripts/plot_synthetic_original.py
scripts/plot_synthetic_alpha_sweep.py
```

FedFisher aggregation remains on the original implementation path:

```text
run_one_shot_algs.py
algs/fisher_avg.py
utils/compress_fisher.py
```

## Install

Clone the repository and install the minimal dependencies in your own Python
environment:

```bash
git clone git@github.com:Ziqi-Gao/FedFisher.git
cd FedFisher
pip install -r requirements.txt
```

The checked-in code does not include local datasets, generated outputs, logs,
or cluster-specific paths.

## Main Arguments

Required arguments:

- `--dataset`: `SyntheticBinary` or `LocalBinaryCSV`.
- `--model`: `SyntheticMLP` or `SyntheticMLPDeep`.
- `--algs_to_run`: one or more algorithms. The synthetic pipeline is intended
  for `fedavg`, `fedfisher_diag`, and `fedfisher_kfac`.

Useful optional arguments:

- `--seed`: random seed, default `0`.
- `--alpha`: Dirichlet concentration for non-IID splitting, default `0.1`.
- `--num_clients`: number of clients, default `5`.
- `--num_rounds`: local-training and aggregation rounds, default `1`.
- `--local_epochs`: local client epochs, default `30`.
- `--synthetic_split`: `iid` or `noniid`, default `noniid`.
- `--synthetic_num_train`: training examples, default `10000`.
- `--synthetic_num_test`: test examples, default `10000`.
- `--synthetic_dim`: feature dimension, default `100`.
- `--synthetic_signal_dim`: number of informative coordinates, default `10`.
- `--synthetic_signal_strength`: class-mean signal norm, default `0.7`.
- `--synthetic_noise_std`: Gaussian noise standard deviation, default `1.0`.

Synthetic sanity-check example:

```bash
python main.py \
  --dataset SyntheticBinary \
  --model SyntheticMLP \
  --synthetic_split noniid \
  --alpha 0.1 \
  --local_epochs 30 \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

## Using Local Data

Use `LocalBinaryCSV` when you clone the repository and want to run the pipeline
on your own local data. The loader expects numeric CSV files:

- labels must be encoded as `0` and `1`
- all non-label, non-client columns are used as model features
- train and test feature columns must match
- local data files can live anywhere; putting them under `data/` keeps them
  ignored by Git

If your training CSV already has a client column, pass it with
`--local_client_col`. The pipeline will use those groups as federated clients.
The test CSV should contain the same feature columns and label column, without
the client column.

```bash
python main.py \
  --dataset LocalBinaryCSV \
  --local_train_csv data/my_train.csv \
  --local_test_csv data/my_test.csv \
  --local_has_header \
  --local_label_col label \
  --local_client_col client_id \
  --model SyntheticMLP \
  --local_epochs 30 \
  --output_dir results/my_local_run \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

If your training CSV does not have client IDs, the pipeline will create clients
from the global training set. Use IID partitioning:

```bash
python main.py \
  --dataset LocalBinaryCSV \
  --local_train_csv data/my_train.csv \
  --local_test_csv data/my_test.csv \
  --local_has_header \
  --local_label_col label \
  --local_partition iid \
  --num_clients 5 \
  --model SyntheticMLP \
  --output_dir results/my_local_iid \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

Or use Dirichlet non-IID partitioning controlled by `--alpha`:

```bash
python main.py \
  --dataset LocalBinaryCSV \
  --local_train_csv data/my_train.csv \
  --local_test_csv data/my_test.csv \
  --local_has_header \
  --local_label_col label \
  --local_partition noniid \
  --alpha 0.1 \
  --num_clients 5 \
  --model SyntheticMLPDeep \
  --output_dir results/my_local_noniid \
  --algs_to_run fedavg fedfisher_diag fedfisher_kfac
```

For CSVs without a header, omit `--local_has_header` and pass a zero-based
column index. The default label column is `-1`, meaning the last column.

## Slurm Runs

Small run:

```bash
sbatch scripts/run_synthetic_original_fedfisher.slurm
```

1000-seed IID/non-IID run:

```bash
sbatch scripts/run_synthetic_original_fedfisher_1000seeds.slurm
```

1000-seed alpha sweep:

```bash
sbatch scripts/run_synthetic_original_fedfisher_alpha_sweep_1000seeds.slurm
```

The Slurm scripts intentionally omit account and partition directives. Pass
cluster-specific settings with `sbatch --account ... --partition ...` or your
site defaults.

## Results

Outputs are written under:

```text
synthetic_binary_experiment/outputs/original_fedfisher*
```

Logs are written under:

```text
synthetic_binary_experiment/logs/original_fedfisher*
```

Summarize and plot:

```bash
python scripts/summarize_synthetic_original.py \
  --input-dir synthetic_binary_experiment/outputs/original_fedfisher \
  --output synthetic_binary_experiment/outputs/original_fedfisher/summary.csv

python scripts/plot_synthetic_original.py
python scripts/plot_synthetic_alpha_sweep.py
```

## Notes

- Generated outputs, logs, caches, and `__pycache__` files should not be
  committed unless explicitly requested.
- Keep FedFisher implementation changes separate from dataset/model plumbing.
- `algs/fisher_avg.py` and `utils/compress_fisher.py` are the FedFisher
  implementation files to keep stable when packaging this synthetic pipeline.
