import torch
import torch.nn as nn

import config
from utils import init_weights


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=not use_norm)
        self.norm = nn.InstanceNorm2d(
            out_channels, affine=True) if use_norm else None
        self.act = nn.LeakyReLU(.2, True)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        return self.act(out)


class Critic(nn.Module):
    def __init__(self, channels, features):
        super().__init__()
        self.disc = nn.Sequential(
            # 64x64
            Block(channels, features, 4, 2, 1, use_norm=False),  # 32x32
            Block(features, features*2, 4, 2, 1),  # 16x16
            Block(features*2, features*4, 4, 2, 1),  # 8x8
            Block(features*4, features*8, 4, 2, 1),  # 4x4
            nn.Conv2d(features*8, 1, 4, 2, 0),  # 1x1
        )

    def forward(self, x):
        return self.disc(x)


def test():
    x = torch.randn((1, config.CHANNELS)+(config.IMAGE_SIZE,)*2)
    print(x.shape)
    disc = Critic(3, 64)
    init_weights(disc)
    y = disc(x)
    print(y.shape)
    print(y.reshape(-1).shape)


if __name__ == '__main__':
    test()
