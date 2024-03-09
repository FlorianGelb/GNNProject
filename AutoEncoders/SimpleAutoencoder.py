import torch
import numpy as np
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size, bottleneck_size, hidden_sizes, layers):
        super().__init__()
        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.hidden_sizes = hidden_sizes
        self.layers = layers
        self.hidden_layers_encoder = []
        self.hidden_layers_decoder = []

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.output_layer = nn.Linear(hidden_sizes[0], input_size)
        self.bottleneck_layer = nn.Linear(hidden_sizes[-1], bottleneck_size)

        self.hidden_layers_decoder.append(nn.Linear(bottleneck_size,hidden_sizes[-1]))
        self.hidden_layers_decoder.append(nn.ReLU())
        for i in range(1, layers):
            self.hidden_layers_encoder.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.hidden_layers_encoder.append(nn.ReLU())
        for i in reversed(range(1, layers)):
            self.hidden_layers_decoder.append(nn.Linear(hidden_sizes[i], hidden_sizes[i-1]))
            self.hidden_layers_decoder.append(nn.ReLU())

        #self.hidden_layers_encoder.append(nn.Linear(hidden_size, bottleneck_size))

        self.encoder = nn.Sequential(self.input_layer,
                                     nn.ReLU(),
                                     *self.hidden_layers_encoder,
                                     self.bottleneck_layer)

        self.decoder = nn.Sequential(
                                    *self.hidden_layers_decoder,
                                     self.output_layer)

        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))

    def calculate_loss(self, X):
        error_type = nn.MSELoss()
        reconstructed = self.forward(X)
        loss = error_type(reconstructed, X)
        return loss

    def train_autoencoder(self, X_train, ts_size, epochs, lr=.01):
        indices = torch.randperm(len(X_train))[:ts_size]
        X_train = X_train[indices]
        error_type = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        for epoch in range(epochs):
            loss = self.calculate_loss(X_train)
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self, losses