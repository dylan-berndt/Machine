import torch
import torch.nn as nn

from .config import Config


class PatchEmbed(nn.Module):
    def __init__(self, imageSize, patchSize=8, inChannels=3, embedDim=256):
        super().__init__()
        self.numPatches = (imageSize // patchSize) ** 2
        self.proj = nn.Conv2d(inChannels, embedDim, kernel_size=patchSize, stride=patchSize)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchDecode(nn.Module):
    def __init__(self, imageSize, patchSize=8, inChannels=3, embedDim=256):
        super().__init__()
        self.patchGrid = imageSize // patchSize
        self.numPatches = self.patchGrid ** 2
        self.proj = nn.ConvTranspose2d(embedDim, inChannels, kernel_size=patchSize, stride=patchSize)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.patchGrid, self.patchGrid)
        x = self.proj(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.patch = PatchEmbed(config.imageSize, config.patchSize, embedDim=config.embedDim)
        self.pos = nn.Parameter(torch.zeros(1, self.patch.numPatches, config.embedDim))
        self.time = nn.Embedding(config.numTimesteps, config.embedDim)

        encoderLayer = nn.TransformerEncoderLayer(
            d_model=config.embedDim, nhead=config.numHeads, dim_feedforward=config.embedDim * 4, 
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoderLayer, num_layers=config.encoderLayers
        )

        self.norm = nn.LayerNorm(config.embedDim)

        self.decode = PatchDecode(config.imageSize, config.patchSize, embedDim=config.embedDim)

    def forward(self, image, t):
        x = self.patch(image)
        x = x + self.pos

        time = self.time(t)
        x = torch.cat([x, time.unsqueeze(1)], dim=1)

        x = self.norm(self.encoder(x))[:, :-1]
        return self.decode(x)