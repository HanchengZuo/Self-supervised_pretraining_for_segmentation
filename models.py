import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16
from torchvision.transforms import functional as F


class Decoder(nn.Module):
    """
    MLP decoder for segmentation task

    Args:
        latent_dim (int): embedding vector dimension in ViT
        hidden_dim (int): hidden non-linear dimension
        output_dim (int): task output dimension
    """

    def __init__(self, latent_dim:int, hidden_dim:int, output_dim:int):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class MaskedAutoEncoder(nn.Module):
    """
    Masked autoencoder for segmentation task

    Args:
        encoder (nn.Moudule): embedding vector encoder
        decoder (nn.Moudule): task output decoder
    """

    def __init__(self, encoder:nn.Module, decoder:nn.Module):
        super(MaskedAutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        x_masked = mask * x
        encode_patch = self.encoder(x_masked)
        reconstruct = self.decoder(encode_patch)

        return reconstruct
