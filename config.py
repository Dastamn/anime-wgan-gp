from numbers import Number

import torch
from torch.nn.functional import pad
from torchvision import transforms
from typing_extensions import Literal

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = 64
CHANNELS = 3
Z_DIM = 100
EVAL_BATCH = 32

CHECKPOINT_CRITIC = 'critic.pt'
CHECKPOINT_GEN = 'gen.pt'
TB_DIR = 'logs'


class Pad():
    def __init__(self, fill: Number = 0, mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'constant'):
        self.fill = fill
        self.mode = mode

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        h, w = t.shape[-2:]
        pad_h = abs(min(0, h-IMAGE_SIZE))
        pad_w = abs(min(0, w-IMAGE_SIZE))
        return pad(t, pad=(pad_h, pad_w), mode=self.mode, value=self.fill)


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    Pad(mode='reflect'),
    transforms.Normalize([0.5]*CHANNELS, [0.5]*CHANNELS),
])
