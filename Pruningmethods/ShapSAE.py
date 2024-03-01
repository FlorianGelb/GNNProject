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





X = generate_synth_data(10)
X = torch.FloatTensor(X)



input_size = 6
bottleneck_size = 3
hidden_size = 3
layers = 1
model = AutoEncoder(input_size, bottleneck_size,hidden_size,layers)


data_loader = DataLoader(X, batch_size=10, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train
num_epochs = 450
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


def custom_function(model, data, weights, layer, part):
    if part == "encoder":
        weight_matrix_shape = model.encoder[layer].weight.shape
    else:
        weight_matrix_shape = model.decoder[layer].weight.shape
    outputs = []
    for w in weights:
        weight = torch.nn.Parameter(torch.Tensor(w.reshape(weight_matrix_shape)))
        if part == "encoder":
            model.encoder[layer].weight = weight
        else:
            model.decoder[layer].weight = weight
        outputs.append(model.calculate_loss(data).detach().numpy())
    return np.array(outputs)


def prune(model : AutoEncoder, importance_level, data_set, background_data_samples=10):
    link_importance = {}
    for part in [model.encoder, model.decoder]:
        if part == model.encoder:
            key = "encoder"
        else:
            key = "decoder"

        link_importance[key] = {}
        for layer in range(len(part)):
            link_importance[key][layer] = []
            weights_in_layer =part[layer].weight.data.reshape(1, -1).flatten()
            background_data = np.random.uniform(weights_in_layer.min(), weights_in_layer.max(), (background_data_samples, len(weights_in_layer)))
            explainer = shap.KernelExplainer(lambda w: custom_function(model, data_set, w, layer, key), background_data)
            shapley_values = abs(explainer.shap_values(weights_in_layer.detach().numpy()))
            link_importance[key][layer] = shapley_values

    for key, li in link_importance["encoder"].items():
        layer_importance = sum(li)
        sorted_indices = np.flip(np.argsort(li))
        cumulative_importance = np.cumsum(li[sorted_indices])
        cutoff = np.argmax(cumulative_importance >= layer_importance*importance_level)
        mask = np.ones(li.shape,dtype=bool)
        mask[:] = False
        mask[sorted_indices[:cutoff]] = True
        link_importance["encoder"][key] = li * mask







prune(model, 0.5,  X)
