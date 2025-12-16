from mambapy.mamba2 import Mamba2 as Mamba2_
from mambapy.mamba2 import Mamba2Config
from mambapy.mamba2 import ResidualBlock as ResidualBlock_
from torch import nn


class Mamba2(Mamba2_):

    def __init__(self, config: Mamba2Config):
        super(Mamba2_, self).__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)],
        )

    def step(self, x, caches):
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])
        return x, caches


class ResidualBlock(ResidualBlock_):

    def step(self, x, cache):
        out, cache = self.mixer.step(self.norm(x).unsqueeze(1), cache)
        out = out.squeeze(1) + x
        return out, cache
