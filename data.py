import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader


def __getDirichletData__(y, n, alpha, num_c):

        min_size = 0
        N = len(y)
        net_dataidx_map = {}
        p_client = np.zeros((n,num_c))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,num_c))
        idx_batch = [[] for _ in range(n)]

        for k in range(num_c):
            idx_k = np.where(y == k)[0]
            np.random.shuffle(idx_k)
            proportions = p_client[:,k]
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

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

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
    else:
        if split == "iid":
            priors = [0.5] * n_client
        elif split == "mild":
            if n_client != 5:
                raise ValueError("synthetic_split='mild' currently expects num_clients=5")
            priors = [0.30, 0.40, 0.50, 0.60, 0.70]
        else:
            raise ValueError("synthetic_split must be one of: iid, mild, noniid")

        base = num_train // n_client
        remainder = num_train % n_client
        dataset_train = []
        client_counts = {}
        xs = []
        ys = []
        for i, prior in enumerate(priors):
            size = base + (1 if i < remainder else 0)
            x_client, y_client = _sample_synthetic_binary(size, prior, mu, noise_std, rng)
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
        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

    x_test, y_test = _sample_synthetic_binary(num_test, 0.5, mu, noise_std, rng)
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
    seed=0,
):

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
    trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    if(datatype=='SyntheticBinary'):
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

    if(datatype=='CIFAR10' or datatype=='SVHN' or datatype == 'GTSRB' or datatype=='CIFAR100' or datatype =='FashionMNIST' or datatype == 'CINIC10'):
    
        if(datatype=='CIFAR10'):
            dataset_train_global = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)

        if(datatype=='SVHN'):
            dataset_train_global = datasets.SVHN('./data/svhn', split="train",download=True, transform=transforms.Compose([transforms.ToTensor()]))
            dataset_test_global = datasets.SVHN('./data/svhn', split="test",download = True, transform=transforms.Compose([transforms.ToTensor()]))

        if(datatype == 'GTSRB'):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32)), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            dataset_train_global = datasets.GTSRB('./data/gtsrb', split="train",download=True, transform=transform)
            dataset_test_global = datasets.GTSRB('./data/gtsrb', split="test",download = True, transform=transform)

        if(datatype=='CINIC10'):
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            cinic_train_dir = './data/cinic_train'
            cinic_test_dir = './data/cinic_test'
            if not os.path.isdir(cinic_train_dir) and os.path.isdir('./data/train'):
                cinic_train_dir = './data/train'
            if not os.path.isdir(cinic_test_dir) and os.path.isdir('./data/test'):
                cinic_test_dir = './data/test'
            dataset_train_global = datasets.ImageFolder(cinic_train_dir,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
            dataset_test_global = datasets.ImageFolder(cinic_test_dir,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

        elif(datatype=='CIFAR100'):
            dataset_train_global = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)

        elif(datatype=='FashionMNIST'):
            dataset_train_global = datasets.FashionMNIST('./data/fashionmnist', train=True, download=True, transform=trans_fashionmnist)
            dataset_test_global = datasets.FashionMNIST('./data/fashionmnist', train=False, download=True, transform=trans_fashionmnist)

        train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
        test_loader  = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))
        X_train = next(iter(train_loader))[0].numpy()
        Y_train = next(iter(train_loader))[1].numpy()
        inds, net_cls_counts = __getDirichletData__(Y_train, n_client, alpha, n_c)
        dataset_train=[]
        for (i,ind) in enumerate(inds):

            ind = inds[i]
            x = X_train[ind]
            y = Y_train[ind]
            x_train = torch.Tensor(x)
            y_train = torch.LongTensor(y)

            print ("Client ", i, " Training examples: " , len(x_train))
            dataset_train_torch = TensorDataset(x_train,y_train)
            dataset_train.append(dataset_train_torch)
    
    return dataset_train, dataset_train_global, dataset_test_global, net_cls_counts
    







    

    




