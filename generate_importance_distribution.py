import torch
import numpy as np
from AutoEncoders.SimpleAutoencoder import AutoEncoder
import Pruning.ShapSAE as SSAE
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from DataPreparation.CustomDataSet import CustomDataSet
from torchvision import datasets

if __name__ == '__main__':

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




    X = torch.FloatTensor(generate_synth_data(15000))
    input_size = 6
    bottleneck_size = 3
    hidden_sizes = [3]
    layers = 0
    model = AutoEncoder(input_size, bottleneck_size,hidden_sizes,layers)


    data_loader = DataLoader(X, batch_size=64, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Train
    num_epochs = 100
    loss_list=[]
    state_dicts = {}#Storage parameters
    for epoch in tqdm(range(num_epochs)):
        for data in data_loader:
            inputs = data
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


    importances = SSAE.calc_importance(model, inputs, batch_size=800, background_data_samples=2)
    plt.matshow(importances[0].reshape(3, 6))
    plt.xticks([0,1,2,3,4,5], ["x1", "x2", "x3", "x4", "x5", "x6"])
    plt.yticks([0,1,2], ["z1", "z2", "z3"])
    plt.colorbar()
    plt.show()
