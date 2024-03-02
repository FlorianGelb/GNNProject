import copy

import torch
import tqdm
import numpy as np
import shap
from AutoEncoders.SimpleAutoencoder import AutoEncoder
from tqdm import tqdm



def custom_function(model, data, weights, layer, node):
    weight_matrix_shape = model.encoder[layer].weight.data[:, node].shape
    outputs = []
    weight_matrix = model.encoder[layer].weight.data
    for w in weights:
        weight_matrix[:, node] = torch.Tensor(w.reshape(weight_matrix_shape))
        model.encoder[layer].weight =  torch.nn.Parameter(weight_matrix)
        outputs.append(model.calculate_loss(data).detach().numpy())
    return np.array(outputs)


def prune(model : AutoEncoder, importance_level, data_set, background_data_samples=1):

    model = copy.deepcopy(model)
    for layer in tqdm(range(len(model.encoder))):
        #if model.encoder[layer]
        if type(model.encoder[layer]) != torch.nn.modules.linear.Linear:
            continue
        encoder_weight_matrix = model.encoder[layer].weight.data
        decoder_weight_matrix = model.decoder[layer].weight.data
        for node in range(model.encoder[layer].weight.shape[1]):
            weights_in_layer = model.encoder[layer].weight.data[:, node].reshape(1, -1).flatten()
            background_data = np.random.uniform(weights_in_layer.min(), weights_in_layer.max(), (background_data_samples, len(weights_in_layer)))
            explainer = shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer, node), background_data)
            shapley_values = abs(explainer.shap_values(weights_in_layer.detach().numpy(), nsamples=len(weights_in_layer)+100))
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
            decoder_layer = len(model.decoder) - layer - 1
            new_weight = torch.nn.Parameter(model.decoder[layer].weight.data[:, node] * mask.T)
            decoder_weight_matrix[:, node] = torch.Tensor(new_weight)
            model.decoder[decoder_layer].weight = torch.nn.Parameter(decoder_weight_matrix)

    return model



