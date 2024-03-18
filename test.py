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


    def transform(input):
        input = torch.FloatTensor(np.array(input))
        input = input.flatten()
        input = input.type(torch.FloatTensor)
        input -= torch.min(input)
        input /= torch.max(input)
        return input

    #X = CustomDataSet("DataPreparation/CorruptedFashionMNIST/Names.csv",
    #                  "DataPreparation/CorruptedFashionMNIST", transform=transform)

    X = datasets.FashionMNIST("/FashionMNIST/", download=False,train=True,transform=transform)
    X = torch.utils.data.Subset(X, range(1000))

    #X = torch.FloatTensor(generate_synth_data(10))
    input_size = 28*28
    bottleneck_size = 25
    hidden_sizes = [40]
    layers = len(hidden_sizes)
    model = AutoEncoder(input_size, bottleneck_size,hidden_sizes,layers)


    data_loader = DataLoader(X, batch_size=512, shuffle=True)

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

    import Pruning.LookaheadPruning as LAP

    plt.plot(range(num_epochs), loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.imshow(model.forward(inputs[0]).detach().numpy().reshape(28,28))
    plt.show()
    plt.imshow(inputs[0].detach().numpy().reshape(28,28))
    plt.show()
    plt.imshow(model.encoder[0].weight.data.detach().numpy())
    plt.show()
    print(model.calculate_loss(inputs))


    importances = SSAE.calc_importance(model, inputs, batch_size=800, background_data_samples=2)
    plt.matshow(importances[0].reshape(-1,40))
    plt.show()


    print(model.calculate_loss(inputs))

    pm, masks = SSAE.prune(model, importances, 0.99)
    pm2, masks = SSAE.prune(model, importances, 0.9)
    #pm2 = SSAE.prune2(model, 999995, inputs, batch_size=800, background_data_samples=3)
    plt.matshow(masks[0].reshape(-1, 40))
    plt.show()
    LAP.apply_lap(model, 2*[0.15])

    plt.imshow(pm.forward(inputs[0]).detach().numpy().reshape(28,28))
    plt.show()
    plt.imshow(pm.encoder[0].weight.data.detach().numpy())
    plt.show()
    print(model.calculate_loss(inputs))
    print(pm.calculate_loss(inputs))
    print(pm2.calculate_loss(inputs))
    #print(sl)
    #print(pm2.calculate_loss(inputs))