import copy

import torch
from torch.nn.utils import vector_to_parameters

from algs.fisher_avg import one_shot_fisher_diag, one_shot_fisher_kfac
from utils.compress_fisher import compress_fisher_kfac, quantize_layer


def get_fedavg_model(d, n, p, args, net_glob, model_vectors):
    fedavg_model = copy.deepcopy(net_glob)
    model_avg = torch.zeros(d).to(args["device"])
    for i in range(n):
        model_avg += p[i] * model_vectors[i]
    vector_to_parameters(model_avg, fedavg_model.parameters())
    return fedavg_model


def get_fisher_merge_model(d, n, p, args, net_glob, model_vectors, F_diag_list):
    grad_diag_sum = torch.zeros(d).to(args["device"])
    F_diag_sum = torch.zeros(d).to(args["device"])
    model_vector_sum = torch.zeros(d).to(args["device"])
    model_avg = torch.zeros(d).to(args["device"])

    for i in range(n):
        model_avg += p[i] * model_vectors[i]

    for i in range(n):
        F_diag_quantized = quantize_layer(F_diag_list[i], net_glob, 15)
        model_vector_quantized = quantize_layer(model_vectors[i], net_glob, 15)
        grad_diag = F_diag_quantized * model_vector_quantized
        grad_diag_sum += grad_diag
        F_diag_sum += F_diag_quantized
        model_vector_sum += p[i] * model_vector_quantized

    ind = torch.where(F_diag_sum < 1e-6)[0].detach().cpu().numpy()
    fisher_avg_model = copy.deepcopy(net_glob)
    fedavg_quantized_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_vector_sum, fedavg_quantized_model.parameters())
    fisher_avg = grad_diag_sum / F_diag_sum
    fisher_avg[ind] = model_avg[ind]
    vector_to_parameters(fisher_avg, fisher_avg_model.parameters())
    return fisher_avg_model


def get_fisher_diag_model(d, n, p, args, net_glob, model_vectors, F_diag_list, dataset_val):
    args_fisher = {}
    args_fisher["eta"] = 0.01
    args_fisher["T"] = 2000
    grad_diag_sum = torch.zeros(d).to(args["device"])
    F_diag_sum = torch.zeros(d).to(args["device"])
    model_vector_sum = torch.zeros(d).to(args["device"])

    for i in range(n):
        F_diag_quantized = quantize_layer(F_diag_list[i], net_glob, 15)
        model_vector_quantized = quantize_layer(model_vectors[i], net_glob, 15)
        grad_diag = F_diag_quantized * model_vector_quantized
        grad_diag_sum += grad_diag
        F_diag_sum += F_diag_quantized
        model_vector_sum += p[i] * model_vector_quantized

    fisher_avg_model = copy.deepcopy(net_glob)
    fedavg_quantized_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_vector_sum, fedavg_quantized_model.parameters())
    fisher_avg = one_shot_fisher_diag(
        fedavg_quantized_model,
        F_diag_sum,
        grad_diag_sum,
        p,
        dataset_val,
        args_fisher,
        args,
    )
    vector_to_parameters(fisher_avg, fisher_avg_model.parameters())
    return fisher_avg_model


def get_fisher_kfac_model(
    d,
    n,
    p,
    args,
    net_glob,
    models,
    model_vectors,
    F_kfac_list,
    dataset_val,
    dataset_train,
):
    args_fisher = {}
    args_fisher["eta"] = 0.01
    args_fisher["lambda"] = 0
    args_fisher["T"] = 2000
    model_vector_sum = torch.zeros(d).to(args["device"])

    for i in range(n):
        model_vector_quantized = quantize_layer(model_vectors[i], net_glob, 15)
        model_vector_sum += p[i] * model_vector_quantized

    fedavg_quantized_model = copy.deepcopy(net_glob)
    vector_to_parameters(model_vector_sum, fedavg_quantized_model.parameters())
    F_kfac_list_comp, grad_avg_comp = compress_fisher_kfac(
        args,
        F_kfac_list,
        models,
        model_vectors,
        dataset_train,
        p,
    )
    fisher_avg_model = copy.deepcopy(net_glob)
    fisher_avg = one_shot_fisher_kfac(
        fedavg_quantized_model,
        F_kfac_list_comp,
        grad_avg_comp,
        p,
        dataset_val,
        args_fisher,
        args,
    )
    vector_to_parameters(fisher_avg, fisher_avg_model.parameters())
    return fisher_avg_model


def get_one_shot_model(
    alg,
    d,
    n,
    p,
    args,
    net_glob,
    models,
    model_vectors,
    F_kfac_list,
    F_diag_list,
    dataset_val,
    dataset_train,
    dataset_train_global,
    dataset_test_global,
    filename,
    net_cls_counts,
):
    if alg == "fedavg":
        return get_fedavg_model(d, n, p, args, net_glob, model_vectors)
    if alg == "fisher_merge":
        return get_fisher_merge_model(d, n, p, args, net_glob, model_vectors, F_diag_list)
    if alg == "fedfisher_diag":
        return get_fisher_diag_model(d, n, p, args, net_glob, model_vectors, F_diag_list, dataset_val)
    if alg == "fedfisher_kfac":
        return get_fisher_kfac_model(
            d,
            n,
            p,
            args,
            net_glob,
            models,
            model_vectors,
            F_kfac_list,
            dataset_val,
            dataset_train,
        )
    raise ValueError("Unsupported one-shot algorithm: %s" % alg)
