import os
import re
from glob import glob
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.optim.optimizer import Optimizer

import config


def init_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.normal_(m.weight.data, mean=.0, std=.02)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    b, c, h, w = real.shape
    eps = torch.rand((b, 1, 1, 1)).repeat((1, c, h, w)).to(config.DEVICE)
    inter_im = real*eps + fake*(1-eps)
    score = critic(inter_im)
    grad = torch.autograd.grad(inputs=inter_im, outputs=score,
                               grad_outputs=torch.ones_like(score),
                               create_graph=True, retain_graph=True)[0]
    grad_samples = grad.shape[0]
    grad = grad.view(grad_samples, -1)
    norm = grad.norm(2, dim=1)
    return torch.mean((norm-1)**2)


def load_checkpoint(model: nn.Module, optimizer: Optimizer, lr: float, filename: str, dir: str = 'checkpoint') -> Tuple[int, int]:
    if dir:
        filename = os.path.join(dir, filename)
    print(f"=> Loading checkpoint from '{filename}'...")
    try:
        checkpoint = torch.load(filename, map_location=config.DEVICE)
    except:
        print('No checkpoint found.')
        return 1, 0
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Done.')
    batch = checkpoint['batch'] if 'batch' in checkpoint else float('inf')
    return checkpoint['epoch'], batch


def save_checkpoint(model: nn.Module, optimizer: Optimizer, filename: str, epoch: int, batch: int = None, dir: str = 'checkpoint'):
    if dir:
        check_dir(dir)
        filename = os.path.join(dir, filename)
    print(f"=> Saving checkpoint to '{filename}'...")
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    if batch != None:
        checkpoint['batch'] = batch
    torch.save(checkpoint, filename)
    print('Done.')


def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_gif(src_dir: str, filename: str = None):
    assert os.path.isdir(src_dir), 'Directory not found.'
    files = sorted(glob(f'{src_dir}/*.jpg'), key=_get_batch)
    assert len(files) > 1, 'Must provide more than 1 file.'
    frames = [Image.open(img) for img in files]
    _, dir = os.path.split(src_dir)
    if not filename:
        filename = f'{dir}.gif'
    elif not filename.endswith('.gif'):
        filename += '.gif'
    frames[0].save(filename, format='gif',
                   append_images=frames[1:], save_all=True, loop=0)


def _get_batch(x: str):
    n = re.findall(r'\d+', x)[:2]
    return int(n[0]), int(n[1])
