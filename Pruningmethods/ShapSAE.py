import copy

import torch
import tqdm
import numpy as np
import shap
from AutoEncoders.SimpleAutoencoder import AutoEncoder
from tqdm import tqdm
from multiprocessing import Pool

import copy
import numpy as np
import torch
import shap
from tqdm import tqdm
from multiprocessing import Pool


def custom_function(model, data, weights, layer, nodes):
    outputs = []
    weight_matrix = model.encoder[layer].weight.data.clone()
    for w in weights:
        for node in nodes:
            weight_matrix[:, node] = torch.Tensor(w)
        model.encoder[layer].weight = torch.nn.Parameter(weight_matrix)
        outputs.append(model.calculate_loss(data).detach().numpy())
    return np.array(outputs)


def compute_shap_values(args):
    model, data_set, weights, layer, nodes = args
    return shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer, nodes), weights).shap_values(
        weights)


def parallel_map_with_progress(pool, func, iterable, total=None):
    results = []
    with tqdm(total=total) as pbar:
        for result in pool.imap(func, iterable):
            results.append(result)
            pbar.update()
    return results


def prune(model, importance_level, data_set, background_data_samples=5, batch_size=10, n_jobs=None):
    model = copy.deepcopy(model)
    for layer in tqdm(range(len(model.encoder))):
        if type(model.encoder[layer]) != torch.nn.modules.linear.Linear:
            continue
        encoder_weight_matrix = model.encoder[layer].weight.data
        decoder_layer = len(model.decoder) - layer - 1
        decoder_weight_matrix = model.decoder[decoder_layer].weight.data

        # Generate background data for SHAP
        weights_in_layer = model.encoder[layer].weight.data.flatten()
        background_data = np.random.uniform(weights_in_layer.min(), weights_in_layer.max(),
                                            (background_data_samples, len(weights_in_layer)))

        # Parallelize computation of SHAP values for nodes in the layer
        if n_jobs == -1:
            processes = None  # Use all available CPU cores
        else:
            processes = n_jobs

        pool = Pool(processes=processes)

        total_nodes = model.encoder[layer].weight.shape[1]
        num_batches = (total_nodes + batch_size - 1) // batch_size

        results = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_nodes)
            nodes_batch = list(range(start, end))

            results.extend(parallel_map_with_progress(pool, compute_shap_values,
                                                      [(model, data_set, background_data, layer, nodes_batch)]))

        pool.close()
        pool.join()

        for node, shap_values in enumerate(results):
            shapley_values = abs(shap_values)
            layer_importance = sum(shapley_values)
            sorted_indices = np.flip(np.argsort(shapley_values))
            cumulative_importance = np.cumsum(shapley_values[sorted_indices])
            cutoff = np.argmax(cumulative_importance >= layer_importance * importance_level)
            mask = np.ones_like(shapley_values, dtype=bool)
            mask[:] = False
            mask[sorted_indices[:cutoff]] = True
            mask = mask.reshape(encoder_weight_matrix[:, node].shape)
            new_weight = torch.nn.Parameter(encoder_weight_matrix[:, node] * mask)
            encoder_weight_matrix[:, node] = torch.Tensor(new_weight)
            model.encoder[layer].weight = torch.nn.Parameter(encoder_weight_matrix)
            new_weight = torch.nn.Parameter(decoder_weight_matrix[node, :] * mask)
            decoder_weight_matrix[node, :] = torch.Tensor(new_weight)
            model.decoder[decoder_layer].weight = torch.nn.Parameter(decoder_weight_matrix)

        break

    return model



"""
def custom_function(model, data, weights, layer, node):
    weight_matrix_shape = model.encoder[layer].weight.data[:, node].shape
    outputs = []
    weight_matrix = model.encoder[layer].weight.data
    for w in weights:
        weight_matrix[:, node] = torch.Tensor(w.reshape(weight_matrix_shape))
        model.encoder[layer].weight =  torch.nn.Parameter(weight_matrix)
        outputs.append(model.calculate_loss(data).detach().numpy())
    return np.array(outputs)


def prune(model : AutoEncoder, importance_level, data_set, background_data_samples=5):

    model = copy.deepcopy(model)
    for layer in tqdm(range(len(model.encoder))):
        #if model.encoder[layer]
        if type(model.encoder[layer]) != torch.nn.modules.linear.Linear:
            continue
        encoder_weight_matrix = model.encoder[layer].weight.data
        decoder_layer = len(model.decoder) - layer - 1
        decoder_weight_matrix = model.decoder[decoder_layer].weight.data
        for node in tqdm(range(model.encoder[layer].weight.shape[1])):
            weights_in_layer = model.encoder[layer].weight.data[:, node].reshape(1, -1).flatten()
            background_data = np.random.uniform(weights_in_layer.min(), weights_in_layer.max(), (background_data_samples, len(weights_in_layer)))
            explainer = shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer, node), background_data)
            shapley_values = abs(explainer.shap_values(weights_in_layer.detach().numpy()))
            layer_importance = sum(shapley_values)
            sorted_indices = np.flip(np.argsort(shapley_values))
            cumulative_importance = np.cumsum(shapley_values[sorted_indices])
            cutoff = np.argmax(cumulative_importance >= layer_importance * importance_level)
            mask = np.ones(shapley_values.shape, dtype=bool)
            mask[:] = False
            mask[sorted_indices[:cutoff]] = True
            mask = mask.reshape(model.encoder[layer].weight.data[:, node].shape)
            new_weight = torch.nn.Parameter(model.encoder[layer].weight.data[:, node] * mask)
            encoder_weight_matrix[:, node] = torch.Tensor(new_weight)
            model.encoder[layer].weight = torch.nn.Parameter(encoder_weight_matrix)
            new_weight = torch.nn.Parameter(model.decoder[decoder_layer].weight.data[node, :] * mask)
            decoder_weight_matrix[node, :] = torch.Tensor(new_weight)
            model.decoder[decoder_layer].weight = torch.nn.Parameter(decoder_weight_matrix)
        break

    return model

"""

