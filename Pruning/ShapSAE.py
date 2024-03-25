import copy

import torch
import tqdm
import numpy as np
import shap
from AutoEncoders.SimpleAutoencoder import AutoEncoder
from tqdm import tqdm
import multiprocessing

import copy
import numpy as np
import torch
import shap
from tqdm import tqdm
from multiprocessing import Pool


def custom_function(model, data, weights, layer, index_start, index_stop):
    weight_matrix_shape =  model.encoder[layer].weight.data.shape
    weight_matrix = model.encoder[layer].weight.data.reshape(1, -1).flatten()
    original_weight_matrix = torch.nn.Parameter(copy.deepcopy(weight_matrix.reshape(weight_matrix_shape)))
    outputs = []
    for w in weights:
        weight_matrix[index_start:index_stop] = torch.FloatTensor(w)
        weight = torch.nn.Parameter(weight_matrix.reshape(weight_matrix_shape))
        model.encoder[layer].weight = weight
        outputs.append(model.calculate_loss(data).detach().numpy())
        model.encoder[layer].weight = original_weight_matrix
    return np.array(outputs)


def calc_importance(model: AutoEncoder, data_set, batch_size=400, background_data_samples=3):
    importance = {}
    for layer in tqdm(reversed(range(len(model.encoder)))):
        if type(model.encoder[layer]) is not torch.nn.modules.linear.Linear:
            continue
        weights_in_layer = model.encoder[layer].weight.data.reshape(1, -1).flatten()

        num_batches = int(np.ceil(len(weights_in_layer) / batch_size))
        shapley_values = np.array([])

        for i in tqdm(range(num_batches)):
            index_start, index_stop = i * batch_size, min((i + 1) * batch_size, len(weights_in_layer))
            batch_weights = weights_in_layer.detach().numpy()[index_start:index_stop]
            background_data = np.random.uniform(batch_weights.min(), batch_weights.max(),
                                                (background_data_samples,  index_stop - index_start))
            explainer = shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer,
                                                                       index_start, index_stop), background_data)
            shapley_values_badge = abs(explainer.shap_values(batch_weights))
            shapley_values = np.hstack((shapley_values, shapley_values_badge.flatten()))

        importance[layer] = shapley_values
    return importance


def prune(model, importance, sparsity_level):
    sparsity_level = 1 - sparsity_level

    p_model = copy.deepcopy(model)
    masks = {}
    for layer in tqdm(reversed(range(len(model.encoder)))):
        if type(model.encoder[layer]) != torch.nn.modules.linear.Linear:
            continue
        shapley_values = importance[layer]
        sorted_indices = np.flip(np.argsort(shapley_values))
        cutoff = int(sparsity_level * len(shapley_values))
        mask = np.ones(shapley_values.shape, dtype=bool)
        mask[:] = False
        mask[sorted_indices[:cutoff]] = True

        mask = mask.reshape(model.encoder[layer].weight.shape)

        new_weight = torch.nn.Parameter(model.encoder[layer].weight.data * mask)
        p_model.encoder[layer].weight = new_weight
        p_model.decoder[len(p_model.encoder) - layer-1].weight = torch.nn.Parameter(p_model.decoder[len(p_model.encoder) - layer-1].weight.data * mask.T)
        masks[layer] = mask
    return p_model, masks
