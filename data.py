import csv

import numpy as np
import torch
from torch.utils.data import TensorDataset


def __getDirichletData__(y, n, alpha, num_c):
    min_size = 0
    N = len(y)
    net_dataidx_map = {}
    p_client = np.zeros((n, num_c))

    for i in range(n):
        p_client[i] = np.random.dirichlet(np.repeat(alpha, num_c))
    idx_batch = [[] for _ in range(n)]

    for k in range(num_c):
        idx_k = np.where(y == k)[0]
        np.random.shuffle(idx_k)
        proportions = p_client[:, k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

    for j in range(n):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    local_sizes = []
    for i in range(n):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)

    print("Data statistics: %s" % str(net_cls_counts))
    print("Data ratio: %s" % str(weights))

    return idx_batch, net_cls_counts


def _sample_synthetic_binary(num_samples, positive_prior, mu, noise_std, rng):
    num_pos = int(round(num_samples * positive_prior))
    num_pos = min(max(num_pos, 0), num_samples)
    y = np.zeros(num_samples, dtype=np.int64)
    y[:num_pos] = 1
    rng.shuffle(y)

    signs = (2 * y - 1).astype(np.float32)
    noise = rng.normal(loc=0.0, scale=noise_std, size=(num_samples, mu.shape[0])).astype(np.float32)
    x = signs[:, None] * mu[None, :] + noise
    return x.astype(np.float32), y


def _get_synthetic_binary_dataset(
    n_client,
    alpha,
    split,
    num_train,
    num_test,
    dim,
    signal_dim,
    signal_strength,
    noise_std,
    seed,
):
    if signal_dim < 1 or signal_dim > dim:
        raise ValueError("synthetic_signal_dim must be in [1, synthetic_dim]")

    rng = np.random.default_rng(seed)
    mu = np.zeros(dim, dtype=np.float32)
    mu[:signal_dim] = signal_strength / np.sqrt(signal_dim)

    if split == "noniid":
        x_train, y_train = _sample_synthetic_binary(num_train, 0.5, mu, noise_std, rng)
        np.random.seed(seed)
        inds, net_cls_counts = __getDirichletData__(y_train, n_client, alpha, 2)
        dataset_train = []
        for i, ind in enumerate(inds):
            x_client = torch.Tensor(x_train[ind])
            y_client = torch.LongTensor(y_train[ind])
            print("Client ", i, " Training examples: ", len(x_client))
            dataset_train.append(TensorDataset(x_client, y_client))
    elif split == "iid":
        base = num_train // n_client
        remainder = num_train % n_client
        dataset_train = []
        client_counts = {}
        xs = []
        ys = []
        for i in range(n_client):
            size = base + (1 if i < remainder else 0)
            x_client, y_client = _sample_synthetic_binary(size, 0.5, mu, noise_std, rng)
            xs.append(x_client)
            ys.append(y_client)
            y_unique, y_counts = np.unique(y_client, return_counts=True)
            client_counts[i] = {y_unique[j]: y_counts[j] for j in range(len(y_unique))}
            print("Client ", i, " Training examples: ", size)
            dataset_train.append(TensorDataset(torch.Tensor(x_client), torch.LongTensor(y_client)))
        x_train = np.concatenate(xs, axis=0)
        y_train = np.concatenate(ys, axis=0)
        net_cls_counts = client_counts
        weights = np.array([len(dataset) for dataset in dataset_train], dtype=np.float64)
        weights = weights / weights.sum()
        print("Data statistics: %s" % str(net_cls_counts))
        print("Data ratio: %s" % str(weights))
    else:
        raise ValueError("synthetic_split must be one of: iid, noniid")

    x_test, y_test = _sample_synthetic_binary(num_test, 0.5, mu, noise_std, rng)
    dataset_train_global = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    dataset_test_global = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts


def _sample_synthetic_effect_modifier(
    num_samples,
    covariate_dim,
    modifier_dim,
    signal_strength,
    intercept,
    treatment_prob,
    rng,
):
    x_cov = rng.normal(loc=0.0, scale=1.0, size=(num_samples, covariate_dim)).astype(np.float32)
    treatment = rng.binomial(1, treatment_prob, size=num_samples).astype(np.float32)
    interactions = treatment[:, None] * x_cov

    modifier_weights = np.zeros(covariate_dim, dtype=np.float32)
    modifier_weights[:modifier_dim] = signal_strength / np.sqrt(modifier_dim)
    modifier_weights[1:modifier_dim:2] *= -1.0

    logits = intercept + interactions.dot(modifier_weights)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probabilities).astype(np.int64)
    features = np.concatenate([treatment[:, None], x_cov, interactions], axis=1)
    return features.astype(np.float32), y


def _get_synthetic_effect_modifier_dataset(
    n_client,
    alpha,
    split,
    num_train,
    num_test,
    covariate_dim,
    modifier_dim,
    signal_strength,
    intercept,
    treatment_prob,
    seed,
):
    if modifier_dim < 1 or modifier_dim > covariate_dim:
        raise ValueError("effect_modifier_signal_dim must be in [1, effect_modifier_covariate_dim]")
    if treatment_prob <= 0.0 or treatment_prob >= 1.0:
        raise ValueError("effect_modifier_treatment_prob must be between 0 and 1")

    rng = np.random.default_rng(seed)
    if split == "noniid":
        x_train, y_train = _sample_synthetic_effect_modifier(
            num_train,
            covariate_dim,
            modifier_dim,
            signal_strength,
            intercept,
            treatment_prob,
            rng,
        )
        np.random.seed(seed)
        inds, net_cls_counts = __getDirichletData__(y_train, n_client, alpha, 2)
        dataset_train = []
        for i, ind in enumerate(inds):
            x_client = torch.Tensor(x_train[ind])
            y_client = torch.LongTensor(y_train[ind])
            print("Client ", i, " Training examples: ", len(x_client))
            dataset_train.append(TensorDataset(x_client, y_client))
    elif split == "iid":
        base = num_train // n_client
        remainder = num_train % n_client
        dataset_train = []
        client_counts = {}
        xs = []
        ys = []
        for i in range(n_client):
            size = base + (1 if i < remainder else 0)
            x_client, y_client = _sample_synthetic_effect_modifier(
                size,
                covariate_dim,
                modifier_dim,
                signal_strength,
                intercept,
                treatment_prob,
                rng,
            )
            xs.append(x_client)
            ys.append(y_client)
            y_unique, y_counts = np.unique(y_client, return_counts=True)
            client_counts[i] = {y_unique[j]: y_counts[j] for j in range(len(y_unique))}
            print("Client ", i, " Training examples: ", size)
            dataset_train.append(TensorDataset(torch.Tensor(x_client), torch.LongTensor(y_client)))
        x_train = np.concatenate(xs, axis=0)
        y_train = np.concatenate(ys, axis=0)
        net_cls_counts = client_counts
        weights = np.array([len(dataset) for dataset in dataset_train], dtype=np.float64)
        weights = weights / weights.sum()
        print("Data statistics: %s" % str(net_cls_counts))
        print("Data ratio: %s" % str(weights))
    else:
        raise ValueError("synthetic_split must be one of: iid, noniid")

    x_test, y_test = _sample_synthetic_effect_modifier(
        num_test,
        covariate_dim,
        modifier_dim,
        signal_strength,
        intercept,
        treatment_prob,
        rng,
    )
    dataset_train_global = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    dataset_test_global = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts


def _resolve_column(column, header, num_cols, option_name):
    try:
        idx = int(column)
    except ValueError:
        if header is None:
            raise ValueError("%s must be an integer when local_has_header is false" % option_name)
        if column not in header:
            raise ValueError("%s=%s is not present in the CSV header" % (option_name, column))
        idx = header.index(column)

    if idx < 0:
        idx += num_cols
    if idx < 0 or idx >= num_cols:
        raise ValueError("%s index %d is outside the CSV column range" % (option_name, idx))
    return idx


def _read_local_binary_csv(path, label_col, has_header, client_col=None):
    rows = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader) if has_header else None
        for row in reader:
            if row and any(cell.strip() for cell in row):
                rows.append(row)

    if not rows:
        raise ValueError("Local CSV is empty: %s" % path)

    num_cols = len(rows[0])
    for row in rows:
        if len(row) != num_cols:
            raise ValueError("Local CSV has inconsistent column counts: %s" % path)

    label_idx = _resolve_column(label_col, header, num_cols, "local_label_col")
    client_idx = None
    if client_col is not None and client_col != "":
        client_idx = _resolve_column(client_col, header, num_cols, "local_client_col")

    excluded = {label_idx}
    if client_idx is not None:
        excluded.add(client_idx)
    feature_idxs = [idx for idx in range(num_cols) if idx not in excluded]
    if not feature_idxs:
        raise ValueError("Local CSV must contain at least one feature column")

    x = np.array([[float(row[idx]) for idx in feature_idxs] for row in rows], dtype=np.float32)
    y = np.array([int(float(row[label_idx])) for row in rows], dtype=np.int64)
    labels = set(np.unique(y).tolist())
    if not labels.issubset({0, 1}):
        raise ValueError("LocalBinaryCSV labels must be encoded as 0/1")

    client_ids = None
    if client_idx is not None:
        client_ids = np.array([row[client_idx] for row in rows])

    return x, y, client_ids


def _build_tensor_datasets_from_indices(x_train, y_train, idx_batch):
    dataset_train = []
    net_cls_counts = {}
    for i, ind in enumerate(idx_batch):
        x_client = torch.Tensor(x_train[ind])
        y_client = torch.LongTensor(y_train[ind])
        print("Client ", i, " Training examples: ", len(x_client))
        dataset_train.append(TensorDataset(x_client, y_client))
        y_unique, y_counts = np.unique(y_train[ind], return_counts=True)
        net_cls_counts[i] = {y_unique[j]: y_counts[j] for j in range(len(y_unique))}

    weights = np.array([len(dataset) for dataset in dataset_train], dtype=np.float64)
    weights = weights / weights.sum()
    print("Data statistics: %s" % str(net_cls_counts))
    print("Data ratio: %s" % str(weights))
    return dataset_train, net_cls_counts


def _get_local_binary_csv_dataset(
    n_client,
    alpha,
    local_train_csv,
    local_test_csv,
    local_label_col,
    local_has_header,
    local_partition,
    local_client_col,
    seed,
):
    if local_train_csv is None or local_test_csv is None:
        raise ValueError("LocalBinaryCSV requires local_train_csv and local_test_csv")

    x_train, y_train, client_ids = _read_local_binary_csv(
        local_train_csv,
        local_label_col,
        local_has_header,
        client_col=local_client_col,
    )
    x_test, y_test, _ = _read_local_binary_csv(
        local_test_csv,
        local_label_col,
        local_has_header,
        client_col=None,
    )

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train/test feature dimensions differ")

    if client_ids is not None:
        client_values = sorted(np.unique(client_ids).tolist())
        idx_batch = [np.where(client_ids == client_id)[0].tolist() for client_id in client_values]
        dataset_train, net_cls_counts = _build_tensor_datasets_from_indices(x_train, y_train, idx_batch)
    elif local_partition == "noniid":
        np.random.seed(seed)
        idx_batch, net_cls_counts = __getDirichletData__(y_train, n_client, alpha, 2)
        dataset_train = []
        for i, ind in enumerate(idx_batch):
            x_client = torch.Tensor(x_train[ind])
            y_client = torch.LongTensor(y_train[ind])
            print("Client ", i, " Training examples: ", len(x_client))
            dataset_train.append(TensorDataset(x_client, y_client))
    elif local_partition == "iid":
        rng = np.random.default_rng(seed)
        indices = np.arange(len(y_train))
        rng.shuffle(indices)
        idx_batch = [ind.tolist() for ind in np.array_split(indices, n_client)]
        dataset_train, net_cls_counts = _build_tensor_datasets_from_indices(x_train, y_train, idx_batch)
    else:
        raise ValueError("local_partition must be one of: iid, noniid")

    dataset_train_global = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    dataset_test_global = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts


def get_dataset(
    datatype,
    n_client,
    n_c,
    alpha,
    partition_equal=True,
    synthetic_split="noniid",
    synthetic_num_train=10000,
    synthetic_num_test=10000,
    synthetic_dim=100,
    synthetic_signal_dim=10,
    synthetic_signal_strength=0.7,
    synthetic_noise_std=1.0,
    effect_modifier_covariate_dim=100,
    effect_modifier_signal_dim=10,
    effect_modifier_signal_strength=2.0,
    effect_modifier_intercept=0.0,
    effect_modifier_treatment_prob=0.5,
    local_train_csv=None,
    local_test_csv=None,
    local_label_col="-1",
    local_has_header=False,
    local_partition="noniid",
    local_client_col=None,
    seed=0,
):
    if datatype == "SyntheticBinary":
        return _get_synthetic_binary_dataset(
            n_client=n_client,
            alpha=alpha,
            split=synthetic_split,
            num_train=synthetic_num_train,
            num_test=synthetic_num_test,
            dim=synthetic_dim,
            signal_dim=synthetic_signal_dim,
            signal_strength=synthetic_signal_strength,
            noise_std=synthetic_noise_std,
            seed=seed,
        )
    if datatype == "SyntheticEffectModifier":
        return _get_synthetic_effect_modifier_dataset(
            n_client=n_client,
            alpha=alpha,
            split=synthetic_split,
            num_train=synthetic_num_train,
            num_test=synthetic_num_test,
            covariate_dim=effect_modifier_covariate_dim,
            modifier_dim=effect_modifier_signal_dim,
            signal_strength=effect_modifier_signal_strength,
            intercept=effect_modifier_intercept,
            treatment_prob=effect_modifier_treatment_prob,
            seed=seed,
        )
    if datatype == "LocalBinaryCSV":
        return _get_local_binary_csv_dataset(
            n_client=n_client,
            alpha=alpha,
            local_train_csv=local_train_csv,
            local_test_csv=local_test_csv,
            local_label_col=local_label_col,
            local_has_header=local_has_header,
            local_partition=local_partition,
            local_client_col=local_client_col,
            seed=seed,
        )

    raise ValueError("Only SyntheticBinary, SyntheticEffectModifier, and LocalBinaryCSV are supported in this pipeline")
