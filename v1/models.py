# models.py
import torch
import torch.nn as nn
import config

class Encoder(nn.Module):

    def __init__(self, input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2), # Regularization
            
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self, input_dim=config.INPUT_DIM, latent_dim=config.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, input_dim)
        )

    def forward(self, z):
        return self.net(z)


class LocalAutoencoder(nn.Module):
    """
    Composed model used on clients.
    """
    def __init__(self, encoder: Encoder = None, decoder: Decoder = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.decoder = decoder if decoder is not None else Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)