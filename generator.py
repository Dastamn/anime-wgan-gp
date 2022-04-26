import torch
import torch.nn as nn

import config
from utils import init_weights


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels, features):
        super().__init__()
        self.gen = nn.Sequential(
            Block(z_dim, features*16, 4, 1, 0),
            Block(features*16, features*8, 4, 2, 1),
            Block(features*8, features*4, 4, 2, 1),
            Block(features*4, features*2, 4, 2, 1),
            nn.ConvTranspose2d(features*2, channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)


def test():
    z = torch.randn((1, config.Z_DIM, 1, 1))
    gen = Generator(config.Z_DIM, config.CHANNELS, 8)
    init_weights(gen)
    y = gen(z)
    assert y.shape == (1, config.CHANNELS)+(config.IMAGE_SIZE,)*2


if __name__ == '__main__':
    test()
