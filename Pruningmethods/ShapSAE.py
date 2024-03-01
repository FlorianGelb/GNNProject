import copy

import torch
import tqdm
from sklearn.datasets import make_moons
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np
import shap
from AutoEncoders.SimpleAutoencoder import AutoEncoder
from tqdm import tqdm


def generate_synth_data(n):
    x1 = np.array([0 for i in range(n)]).reshape(n, 1)
    x2 = np.random.uniform(0, 1, (n, 1))
    x3 = np.vstack(
        [np.random.uniform(0, 0.25, (int(np.floor(n / 2)), 1)), np.random.uniform(0.75, 1, (int(np.ceil(n / 2)), 1))])
    x4 = np.vstack(
        [np.random.normal(0.25, 0.1, (int(np.floor(n / 2)), 1)), np.random.normal(0.75, 0.1, (int(np.ceil(n / 2)), 1))])
    x5 = x1 + x2
    x6 = x3 + x4
    return np.hstack([x1, x2, x3, x4, x5, x6])





X = generate_synth_data(100)
X = torch.FloatTensor(X)



input_size = 6
bottleneck_size = 3
hidden_size = 3
layers = 6
model = AutoEncoder(input_size, bottleneck_size,hidden_size,layers)


data_loader = DataLoader(X, batch_size=10, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train
num_epochs = 200
loss_list=[]
state_dicts = {}#Storage parameters
for epoch in tqdm(range(num_epochs)):
    for data in data_loader:
        inputs = data

        # Forward
        outputs = model.forward(inputs)
        loss = criterion(outputs, inputs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_list.append(loss.item())
plt.plot(range(num_epochs), loss_list, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


def custom_function(model, data, weights, layer):
    weight_matrix_shape = model.encoder[layer].weight.shape
    outputs = []
    for w in weights:
        weight = torch.nn.Parameter(torch.Tensor(w.reshape(weight_matrix_shape)))
        model.encoder[layer].weight = weight
        outputs.append(model.calculate_loss(data).detach().numpy())
    return np.array(outputs)


def prune(model : AutoEncoder, importance_level, data_set, background_data_samples=10):

    model = copy.deepcopy(model)
    for layer in tqdm(range(len(model.encoder))):
        #if model.encoder[layer]
        if type(model.encoder[layer]) != torch.nn.modules.linear.Linear:
            continue
        weights_in_layer = model.encoder[layer].weight.data.reshape(1, -1).flatten()

        background_data = np.random.uniform(weights_in_layer.min(), weights_in_layer.max(), (background_data_samples, len(weights_in_layer)))
        explainer = shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer), background_data)
        shapley_values = abs(explainer.shap_values(weights_in_layer.detach().numpy()))
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
#        decoder_layer = len(model.decoder) - layer - 1
#        model.decoder[decoder_layer].weight = torch.nn.Parameter(model.decoder[decoder_layer].weight.data * mask.T)
    return model




print(model.calculate_loss(X))
pruned_model = prune(model, 1,  X)
print(pruned_model.calculate_loss(X))
