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



    def transform(input):
        input = torch.FloatTensor(np.array(input))
        input = input.flatten()
        input = input.type(torch.FloatTensor)
        input -= torch.min(input)
        input /= torch.max(input)
        return input

    X = CustomDataSet("DataPreparation/CorruptedFashionMNIST/Names.csv",
                      "DataPreparation/CorruptedFashionMNIST", transform=transform)

    #X = datasets.MNIST("/MNIST", download=True,train=True,transform=transform)
    X = torch.utils.data.Subset(X, range(6000))

    input_size = 28*28
    bottleneck_size = 50
    hidden_sizes = [128, 64]
    layers = 2
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
            inputs, _ = data
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
    fig, axs = plt.subplots(3, 3)
    for i in range(9):
        axs.flatten()[i].set_ylabel(f"$l^1_{i}$")
        im = axs.flatten()[i].imshow(importances[0].reshape(-1, 28, 28)[i], vmin=0, vmax=max(importances[0]),
                                     cmap="gnuplot")
    fig.suptitle("Importance of input in the first layer")
    cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    print(model.calculate_loss(inputs))

    pm, masks = SSAE.prune(model, importances, 0.9)
    pm2, masks1 = SSAE.prune(model, importances, 0.1)

    fig, axs = plt.subplots(3,3)
    for i in range(9):
        axs.flatten()[i].set_ylabel(f"$l^1_{i}$")
        axs.flatten()[i].imshow(masks[0].reshape(-1, 28, 28)[i])
    fig.suptitle("Mask of the first layer at sparsity level 0.9")
    plt.tight_layout()
    plt.show()
    print(model.calculate_loss(inputs))

    plt.imshow(model.encoder[0].weight.data - pm.encoder[0].weight.data)
    plt.colorbar()
    plt.show()