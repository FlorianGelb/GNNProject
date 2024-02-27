import torch
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




# Create training set and testing
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


# convert it to tensor
dataset = TensorDataset(X, X)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)

#Set parameters
input_size=6
bottleneck_size=3
hidden_size=3
layers=1
model = AutoEncoder(input_size, bottleneck_size,hidden_size,layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Train
num_epochs = 1
loss_list=[]
state_dicts = {}#Storage parameters
for epoch in range(num_epochs):
    for data in data_loader:
        inputs, _ = data

        # Forward
        outputs = model.forward(inputs)
        loss = criterion(outputs, inputs)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    loss_list.append(loss.item())


def custom_masker(mask, x):
    return (x * mask)



# Plot loss
#plt.plot(range(num_epochs), loss_list, label='Training Loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

def proxy_function(autoencoder: AutoEncoder, weights, layer, data):
    data = torch.Tensor(data)
    old_weights = autoencoder.encoder[layer].weight.data
    loss = []
    for i in range(len(weights)):
        old_weights[i, 0] = torch.Tensor(weights)[i]
        autoencoder.encoder[layer].weight = torch.nn.Parameter(old_weights)
        loss.append(autoencoder.calculate_loss(data).detach().numpy())
    return loss


print(model.encoder[0].weight.data)
explainer = shap.Explainer(lambda w: proxy_function(model, w, 0, X), custom_masker)
values = explainer(model.encoder[0].weight.data.reshape(-1, 1))
print(values)