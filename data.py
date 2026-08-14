import csv

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset


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
    main_effect_strength,
    confounding_strength,
    intercept,
    treatment_prob,
    propensity_clip,
    rng,
    covariate_mean=0.0,
    covariate_std=1.0,
):
    if covariate_std <= 0.0:
        raise ValueError("covariate_std must be positive")
    x_cov = rng.normal(
        loc=covariate_mean,
        scale=covariate_std,
        size=(num_samples, covariate_dim),
    ).astype(np.float32)

    modifier_weights = np.zeros(covariate_dim, dtype=np.float32)
    modifier_weights[:modifier_dim] = signal_strength / np.sqrt(modifier_dim)
    modifier_weights[1:modifier_dim:2] *= -1.0

    main_effect_weights = np.zeros(covariate_dim, dtype=np.float32)
    main_effect_weights[:modifier_dim] = main_effect_strength / np.sqrt(modifier_dim)

    confounder_weights = np.zeros(covariate_dim, dtype=np.float32)
    confounder_weights[:modifier_dim] = confounding_strength / np.sqrt(modifier_dim)
    propensity_intercept = np.log(treatment_prob / (1.0 - treatment_prob))
    propensity_logits = propensity_intercept + x_cov.dot(confounder_weights)
    propensities = 1.0 / (1.0 + np.exp(-np.clip(propensity_logits, -30.0, 30.0)))
    propensities = np.clip(propensities, propensity_clip, 1.0 - propensity_clip)
    treatment = rng.binomial(1, propensities).astype(np.float32)
    interactions = treatment[:, None] * x_cov

    logits = (
        intercept
        + x_cov.dot(main_effect_weights)
        + interactions.dot(modifier_weights)
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    y = rng.binomial(1, probabilities).astype(np.int64)
    features = np.concatenate([treatment[:, None], x_cov, interactions], axis=1)
    return features.astype(np.float32), y


def _sample_effect_modifier_site_sizes(n_client, min_samples, max_samples, rng):
    if n_client < 1:
        raise ValueError("num_clients must be at least 1")
    if min_samples < 1:
        raise ValueError("effect_modifier_site_min_samples must be at least 1")
    if max_samples < min_samples:
        raise ValueError(
            "effect_modifier_site_max_samples must be greater than or equal to "
            "effect_modifier_site_min_samples"
        )
    return rng.integers(min_samples, max_samples + 1, size=n_client, dtype=np.int64)


def get_effect_modifier_site_gaussian_parameters(
    n_client,
    mean_span=0.75,
    std_low=0.75,
    std_high=1.25,
):
    """Return a deterministic mean-by-scale design for site covariate shift.

    With the default 10 sites this is the Cartesian product of five means
    ``[-0.75, -0.375, 0, 0.375, 0.75]`` and two standard deviations
    ``[0.75, 1.25]``. Every raw covariate at a site uses the same scalar mean
    and standard deviation, so the only structural change is the site's
    Gaussian covariate distribution.
    """
    if n_client < 1:
        raise ValueError("num_clients must be at least 1")
    if mean_span < 0.0:
        raise ValueError("effect_modifier_site_gaussian_mean_span must be nonnegative")
    if std_low <= 0.0 or std_high <= 0.0:
        raise ValueError("site Gaussian standard deviations must be positive")
    if std_high < std_low:
        raise ValueError("site Gaussian std_high must be greater than or equal to std_low")

    mean_count = (n_client + 1) // 2
    if mean_count == 1:
        means = np.array([0.0], dtype=np.float64)
    else:
        means = np.linspace(-mean_span, mean_span, mean_count, dtype=np.float64)
    stds = (float(std_low), float(std_high))
    parameters = [(float(site_mean), site_std) for site_mean in means for site_std in stds]
    return parameters[:n_client]


def _sample_effect_modifier_with_label_targets(
    target_counts,
    covariate_dim,
    modifier_dim,
    signal_strength,
    main_effect_strength,
    confounding_strength,
    intercept,
    treatment_prob,
    propensity_clip,
    rng,
    covariate_mean,
    covariate_std,
):
    """Sample one site while retaining the baseline site's label counts."""
    target_counts = np.asarray(target_counts, dtype=np.int64)
    if target_counts.shape != (2,) or np.any(target_counts < 0):
        raise ValueError("target_counts must contain two nonnegative class counts")

    accepted_features = [[], []]
    accepted_labels = [[], []]
    remaining = target_counts.copy()
    total_target = int(target_counts.sum())
    max_draws = max(100000, 1000 * total_target)
    draws = 0

    while np.any(remaining > 0):
        batch_size = max(256, 4 * int(remaining.sum()))
        features, labels = _sample_synthetic_effect_modifier(
            batch_size,
            covariate_dim,
            modifier_dim,
            signal_strength,
            main_effect_strength,
            confounding_strength,
            intercept,
            treatment_prob,
            propensity_clip,
            rng,
            covariate_mean=covariate_mean,
            covariate_std=covariate_std,
        )
        draws += batch_size
        for label in (0, 1):
            need = int(remaining[label])
            if need == 0:
                continue
            label_indices = np.flatnonzero(labels == label)[:need]
            if label_indices.size:
                accepted_features[label].append(features[label_indices])
                accepted_labels[label].append(labels[label_indices])
                remaining[label] -= label_indices.size
        if draws > max_draws and np.any(remaining > 0):
            raise RuntimeError(
                "site Gaussian rejection sampling could not reach target label counts"
            )

    feature_blocks = []
    label_blocks = []
    for label in (0, 1):
        if accepted_features[label]:
            feature_blocks.append(np.concatenate(accepted_features[label], axis=0))
            label_blocks.append(np.concatenate(accepted_labels[label], axis=0))
    site_features = np.concatenate(feature_blocks, axis=0)
    site_labels = np.concatenate(label_blocks, axis=0)
    order = rng.permutation(total_target)
    return site_features[order], site_labels[order]


def _replace_with_site_gaussian_training_data(
    y_train,
    idx_batch,
    site_parameters,
    covariate_dim,
    modifier_dim,
    signal_strength,
    main_effect_strength,
    confounding_strength,
    intercept,
    treatment_prob,
    propensity_clip,
    seed,
):
    site_features = []
    site_labels = []
    new_indices = []
    offset = 0

    for site_idx, (indices, (site_mean, site_std)) in enumerate(
        zip(idx_batch, site_parameters)
    ):
        target_counts = np.bincount(y_train[indices], minlength=2)
        site_rng = np.random.default_rng(seed + 2000003 + site_idx * 104729)
        features, labels = _sample_effect_modifier_with_label_targets(
            target_counts,
            covariate_dim,
            modifier_dim,
            signal_strength,
            main_effect_strength,
            confounding_strength,
            intercept,
            treatment_prob,
            propensity_clip,
            site_rng,
            covariate_mean=site_mean,
            covariate_std=site_std,
        )
        site_features.append(features)
        site_labels.append(labels)
        new_indices.append(list(range(offset, offset + len(labels))))
        offset += len(labels)

    return (
        np.concatenate(site_features, axis=0),
        np.concatenate(site_labels, axis=0),
        new_indices,
    )


def _partition_indices_with_capacities(y, capacities, alpha, num_c, rng):
    """Create label-heterogeneous sites while exactly respecting capacities."""
    if alpha <= 0.0:
        raise ValueError("alpha must be positive for noniid splitting")

    capacities = np.asarray(capacities, dtype=np.int64)
    if capacities.sum() != len(y):
        raise ValueError("site capacities must sum to the number of training examples")

    n_client = len(capacities)
    client_class_preferences = np.stack(
        [rng.dirichlet(np.repeat(alpha, num_c)) for _ in range(n_client)],
        axis=0,
    )
    remaining = capacities.copy()
    idx_batch = [[] for _ in range(n_client)]
    indices = np.arange(len(y))
    rng.shuffle(indices)

    for sample_idx in indices:
        available = np.flatnonzero(remaining > 0)
        label = int(y[sample_idx])
        weights = client_class_preferences[available, label] * remaining[available]
        if not np.isfinite(weights).all() or weights.sum() <= 0.0:
            weights = remaining[available].astype(np.float64)
        probabilities = weights / weights.sum()
        client_idx = int(rng.choice(available, p=probabilities))
        idx_batch[client_idx].append(int(sample_idx))
        remaining[client_idx] -= 1

    if np.any(remaining != 0):
        raise RuntimeError("capacity-constrained partition did not fill every site")
    return idx_batch


def _get_synthetic_effect_modifier_dataset(
    n_client,
    alpha,
    split,
    num_test,
    covariate_dim,
    modifier_dim,
    signal_strength,
    main_effect_strength,
    confounding_strength,
    intercept,
    treatment_prob,
    propensity_clip,
    site_min_samples,
    site_max_samples,
    site_gaussian_shift,
    site_gaussian_mean_span,
    site_gaussian_std_low,
    site_gaussian_std_high,
    seed,
):
    if modifier_dim < 1 or modifier_dim > covariate_dim:
        raise ValueError("effect_modifier_signal_dim must be in [1, effect_modifier_covariate_dim]")
    if treatment_prob <= 0.0 or treatment_prob >= 1.0:
        raise ValueError("effect_modifier_treatment_prob must be between 0 and 1")
    if propensity_clip < 0.0 or propensity_clip >= 0.5:
        raise ValueError("effect_modifier_propensity_clip must be in [0, 0.5)")
    if signal_strength < 0.0:
        raise ValueError("effect_modifier_signal_strength must be nonnegative")
    if main_effect_strength < 0.0:
        raise ValueError("effect_modifier_main_effect_strength must be nonnegative")
    if confounding_strength < 0.0:
        raise ValueError("effect_modifier_confounding_strength must be nonnegative")

    rng = np.random.default_rng(seed)
    partition_rng = np.random.default_rng(seed + 1000003)
    site_sizes = _sample_effect_modifier_site_sizes(
        n_client,
        site_min_samples,
        site_max_samples,
        rng,
    )
    num_train = int(site_sizes.sum())
    x_train, y_train = _sample_synthetic_effect_modifier(
        num_train,
        covariate_dim,
        modifier_dim,
        signal_strength,
        main_effect_strength,
        confounding_strength,
        intercept,
        treatment_prob,
        propensity_clip,
        rng,
    )

    if split == "noniid":
        inds = _partition_indices_with_capacities(
            y_train,
            site_sizes,
            alpha,
            2,
            partition_rng,
        )
    elif split == "iid":
        indices = np.arange(num_train)
        partition_rng.shuffle(indices)
        split_points = np.cumsum(site_sizes)[:-1]
        inds = [ind.tolist() for ind in np.split(indices, split_points)]
    else:
        raise ValueError("synthetic_split must be one of: iid, noniid")

    site_parameters = None
    if site_gaussian_shift:
        site_parameters = get_effect_modifier_site_gaussian_parameters(
            n_client,
            mean_span=site_gaussian_mean_span,
            std_low=site_gaussian_std_low,
            std_high=site_gaussian_std_high,
        )
        x_train, y_train, inds = _replace_with_site_gaussian_training_data(
            y_train,
            inds,
            site_parameters,
            covariate_dim,
            modifier_dim,
            signal_strength,
            main_effect_strength,
            confounding_strength,
            intercept,
            treatment_prob,
            propensity_clip,
            seed,
        )

    dataset_train, net_cls_counts = _build_tensor_datasets_from_indices(
        x_train,
        y_train,
        inds,
    )
    if site_parameters is not None:
        for site_idx, (site_dataset, (site_mean, site_std)) in enumerate(
            zip(dataset_train, site_parameters)
        ):
            site_dataset.site_idx = site_idx
            site_dataset.site_gaussian_mean = site_mean
            site_dataset.site_gaussian_std = site_std
    x_test, y_test = _sample_synthetic_effect_modifier(
        num_test,
        covariate_dim,
        modifier_dim,
        signal_strength,
        main_effect_strength,
        confounding_strength,
        intercept,
        treatment_prob,
        propensity_clip,
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


def _proportional_site_capacities(total_size, site_sizes):
    site_sizes = np.asarray(site_sizes, dtype=np.float64)
    if total_size < len(site_sizes):
        raise ValueError("local evaluation requires at least one test example per site")
    raw_capacities = total_size * site_sizes / site_sizes.sum()
    capacities = np.floor(raw_capacities).astype(np.int64)
    remainder = int(total_size - capacities.sum())
    order = np.argsort(-(raw_capacities - capacities), kind="mergesort")
    capacities[order[:remainder]] += 1
    if np.any(capacities < 1) or capacities.sum() != total_size:
        raise RuntimeError("could not construct positive local evaluation capacities")
    return capacities


def build_effect_modifier_site_evaluation_datasets(
    dataset_test_global,
    dataset_train,
    split,
    alpha,
    seed,
):
    """Partition one held-out test set into disjoint site-local shards.

    The union of the returned subsets is exactly the original test set. This
    lets global and within-site permutation use identical held-out examples,
    isolating the permutation domain as the only evaluation change.
    """
    site_sizes = [len(site_dataset) for site_dataset in dataset_train]
    capacities = _proportional_site_capacities(len(dataset_test_global), site_sizes)
    rng = np.random.default_rng(seed + 3000017)
    labels = dataset_test_global.tensors[1].detach().cpu().numpy()

    if split == "noniid":
        idx_batch = _partition_indices_with_capacities(
            labels,
            capacities,
            alpha,
            2,
            rng,
        )
    elif split == "iid":
        indices = np.arange(len(dataset_test_global))
        rng.shuffle(indices)
        split_points = np.cumsum(capacities)[:-1]
        idx_batch = [indices.tolist() for indices in np.split(indices, split_points)]
    else:
        raise ValueError("synthetic_split must be one of: iid, noniid")

    site_datasets = []
    for site_idx, indices in enumerate(idx_batch):
        site_dataset = Subset(dataset_test_global, indices)
        site_dataset.site_idx = site_idx
        site_datasets.append(site_dataset)
    return site_datasets


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
    effect_modifier_signal_dim=2,
    effect_modifier_signal_strength=2.0,
    effect_modifier_main_effect_strength=1.0,
    effect_modifier_confounding_strength=1.0,
    effect_modifier_intercept=0.0,
    effect_modifier_treatment_prob=0.5,
    effect_modifier_propensity_clip=0.05,
    effect_modifier_site_min_samples=500,
    effect_modifier_site_max_samples=1000,
    effect_modifier_site_gaussian_shift=False,
    effect_modifier_site_gaussian_mean_span=0.75,
    effect_modifier_site_gaussian_std_low=0.75,
    effect_modifier_site_gaussian_std_high=1.25,
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
            num_test=synthetic_num_test,
            covariate_dim=effect_modifier_covariate_dim,
            modifier_dim=effect_modifier_signal_dim,
            signal_strength=effect_modifier_signal_strength,
            main_effect_strength=effect_modifier_main_effect_strength,
            confounding_strength=effect_modifier_confounding_strength,
            intercept=effect_modifier_intercept,
            treatment_prob=effect_modifier_treatment_prob,
            propensity_clip=effect_modifier_propensity_clip,
            site_min_samples=effect_modifier_site_min_samples,
            site_max_samples=effect_modifier_site_max_samples,
            site_gaussian_shift=effect_modifier_site_gaussian_shift,
            site_gaussian_mean_span=effect_modifier_site_gaussian_mean_span,
            site_gaussian_std_low=effect_modifier_site_gaussian_std_low,
            site_gaussian_std_high=effect_modifier_site_gaussian_std_high,
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
