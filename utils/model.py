import torch
import torch.nn as nn
import math

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
    def __init__(self, imageSize, patchSize=4, inChannels=3, embedDim=256):
        super().__init__()
        self.patchGrid = imageSize // patchSize
        self.patchSize = patchSize
        self.proj = nn.Linear(embedDim, patchSize * patchSize * embedDim // 4)
        self.shuffle = nn.PixelShuffle(patchSize)
        self.head = nn.Sequential(
            nn.Conv2d(embedDim // 4, embedDim // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(embedDim // 4, inChannels, 3, padding=1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = self.proj(x).transpose(1, 2).reshape(B, -1, self.patchGrid, self.patchGrid)
        return self.head(self.shuffle(x))


# class PatchDecode(nn.Module):
#     def __init__(self, imageSize, patchSize=8, inChannels=3, embedDim=256):
#         super().__init__()
#         self.patchGrid = imageSize // patchSize
#         self.numPatches = self.patchGrid ** 2
#         self.proj = nn.ConvTranspose2d(embedDim, inChannels, kernel_size=patchSize, stride=patchSize)

#     def forward(self, x):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).reshape(B, C, self.patchGrid, self.patchGrid)
#         x = self.proj(x)
#         return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Pre-LN transformer block with per-layer AdaLN-Zero conditioning on the
    timestep embedding, so every layer sees the noise level directly instead
    of relying on a single additive vector at the input to survive N layers
    of residual mixing."""

    def __init__(self, embedDim, numHeads, mlpRatio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedDim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embedDim, numHeads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedDim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedDim, embedDim * mlpRatio),
            nn.GELU(),
            nn.Linear(embedDim * mlpRatio, embedDim),
        )

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embedDim, embedDim * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        shiftAttn, scaleAttn, gateAttn, shiftMLP, scaleMLP, gateMLP = self.adaLN(cond).chunk(6, dim=-1)

        h = modulate(self.norm1(x), shiftAttn, scaleAttn)
        attnOut, _ = self.attn(h, h, h, need_weights=False)
        x = x + gateAttn.unsqueeze(1) * attnOut

        h = modulate(self.norm2(x), shiftMLP, scaleMLP)
        x = x + gateMLP.unsqueeze(1) * self.mlp(h)

        return x


class FinalLayer(nn.Module):
    """DiT-style conditioned output norm, zero-initialized so the decode head
    starts out seeing a plain, un-modulated normalized input."""

    def __init__(self, embedDim):
        super().__init__()
        self.norm = nn.LayerNorm(embedDim, elementwise_affine=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embedDim, embedDim * 2))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class ViTEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.patch = PatchEmbed(config.imageSize, config.patchSize, embedDim=config.embedDim)
        self.pos = nn.Parameter(torch.randn(1, self.patch.numPatches, config.embedDim) * 0.2)
        self.timeMLP = nn.Sequential(
            nn.Linear(config.embedDim, config.embedDim * 4),
            nn.SiLU(),
            nn.Linear(config.embedDim * 4, config.embedDim)
        )

        self.blocks = nn.ModuleList([
            DiTBlock(config.embedDim, config.numHeads)
            for _ in range(config.encoderLayers)
        ])

        self.norm = FinalLayer(config.embedDim)

        self.decode = PatchDecode(config.imageSize, config.patchSize, embedDim=config.embedDim)

        nn.init.zeros_(self.decode.proj.weight)
        nn.init.zeros_(self.decode.proj.bias)

    def time(self, t):
        half = self.config.embedDim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, image, t):
        x = self.patch(image)

        cond = self.timeMLP(self.time(t))

        for block in self.blocks:
            x = x + self.pos
            x = block(x, cond)

        x = self.norm(x, cond)
        return self.decode(x)