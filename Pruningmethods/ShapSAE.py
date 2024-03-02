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


def custom_function(model, data, weights, layer, index_start, index_stop):
    weight_matrix_shape =  model.encoder[layer].weight.data.shape
    weight_matrix = model.encoder[layer].weight.data.reshape(1, -1).flatten()
    outputs = []
    for w in weights:
        weight_matrix[index_start:index_stop] = torch.FloatTensor(w)
        weight = torch.nn.Parameter(weight_matrix.reshape(weight_matrix_shape))
        model.encoder[layer].weight = weight
        outputs.append(model.calculate_loss(data).detach().numpy())
    return np.array(outputs)


def prune(model: AutoEncoder, importance_level, data_set, number_of_batch=2000, background_data_samples=10):

    model = copy.deepcopy(model)
    for layer in tqdm(range(len(model.encoder))):
        if type(model.encoder[layer]) != torch.nn.modules.linear.Linear:
            continue
        weights_in_layer = model.encoder[layer].weight.data.reshape(1, -1).flatten()

        num_batches = len(weights_in_layer) // number_of_batch
        shapley_values = []
        background_data = np.random.uniform(weights_in_layer.min(), weights_in_layer.max(),
                                            (background_data_samples, number_of_batch))
        for i in tqdm(range(num_batches)):
            index_start, index_stop = i * number_of_batch, (i + 1) * number_of_batch
            batch_weights = weights_in_layer.detach().numpy()[index_start : index_stop]
            explainer = shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer, index_start, index_stop), background_data)
            shapley_values_badge = abs(explainer.shap_values(batch_weights))
            shapley_values.append(shapley_values_badge)

        shapley_values = np.array(shapley_values).flatten()
        layer_importance = sum(shapley_values)
        sorted_indices = np.flip(np.argsort(shapley_values))
        cumulative_importance = np.cumsum(shapley_values[sorted_indices])
        cutoff = np.argmax(cumulative_importance >= layer_importance * importance_level)

        mask = np.ones(shapley_values.shape, dtype=bool)
        mask[:] = False
        mask[sorted_indices[:cutoff]] = True
        mask = mask.reshape(model.encoder[layer].weight.shape)

        new_weight = torch.nn.Parameter(model.encoder[layer].weight.data * mask)
        model.encoder[layer].weight = new_weight

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

