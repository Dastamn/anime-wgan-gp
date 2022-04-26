import argparse
import os
import shutil

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

import config
from critic import Critic
from generator import Generator
from utils import (check_dir, gradient_penalty, init_weights, load_checkpoint,
                   save_checkpoint)


def train(args: argparse.Namespace):
    # load data
    data = ImageFolder(args.datadir, transform=config.transform)
    loader = DataLoader(data, args.batchsize, shuffle=True)
    # init models
    gen = Generator(config.Z_DIM, config.CHANNELS,
                    features=64).to(config.DEVICE)
    critic = Critic(config.CHANNELS, features=64).to(config.DEVICE)
    # init optimizers
    gen_optim = optim.Adam(gen.parameters(), lr=args.glr, betas=(.0, .9))
    critic_optim = optim.Adam(critic.parameters(), lr=args.clr, betas=(.0, .9))
    # load checkpoints
    start_epoch, start_batch = 1, 0
    if args.resume:
        prev_epoch, prev_batch = load_checkpoint(
            gen, gen_optim, args.glr, config.CHECKPOINT_GEN)
        load_checkpoint(critic, critic_optim, args.clr,
                        config.CHECKPOINT_CRITIC)
        start_epoch = prev_epoch
        if isinstance(start_batch, int):
            start_batch = prev_batch + 1
        if start_batch >= len(loader):
            start_batch = 0
            start_epoch += 1
    else:
        init_weights(gen)
        init_weights(critic)
        shutil.rmtree(config.TB_DIR, ignore_errors=True)
        shutil.rmtree(args.savedir, ignore_errors=True)
    # tensorboard stuff
    noise = torch.randn(config.EVAL_BATCH, config.Z_DIM,
                        1, 1).to(config.DEVICE)
    writer_real = SummaryWriter(f'{config.TB_DIR}/real')
    writer_fake = SummaryWriter(f'{config.TB_DIR}/fake')
    writer_critic = SummaryWriter(f'{config.TB_DIR}/critic')
    step = 0
    # training
    for epoch in range(start_epoch, args.epochs+1):
        print(f'Epoch {epoch}')
        for i, (x, _) in enumerate(tqdm(loader)):
            if epoch == start_epoch and i < start_batch:
                continue
            real = x.to(config.DEVICE)
            # train critic
            for _ in range(args.csteps):
                z = torch.randn((x.shape[0], config.Z_DIM, 1, 1)).to(
                    config.DEVICE)
                fake = gen(z)
                critic_real = critic(real).reshape(-1)  # for 1 dim result
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake)
                critic_loss = (
                    # standard wgan loss
                    - (torch.mean(critic_real)-torch.mean(critic_fake))
                    # gradient penalty
                    + args.lambdagp * gp
                )
                critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_optim.step()
            # train generator
            out = critic(fake).reshape(-1)
            gen_loss = torch.neg(torch.mean(out))
            gen.zero_grad()
            gen_loss.backward()
            gen_optim.step()
            # eval
            if i % 100 == 0:
                print(f'Loss_D: {critic_loss:.4f}, Loss_G: {gen_loss:.4f}')
                with torch.no_grad():
                    # log to tensorboard
                    fake = gen(noise)
                    grid_real = make_grid(
                        real[:config.EVAL_BATCH], normalize=True)
                    grid_fake = make_grid(fake, normalize=True)
                    writer_real.add_image('Real', grid_real, global_step=step)
                    writer_fake.add_image('Fake', grid_fake, global_step=step)
                    writer_critic.add_scalar(
                        'Loss', critic_loss, len(loader)*(epoch-1) + i)
                    # save fakes
                    check_dir(args.savedir)
                    save_image(grid_fake, os.path.join(
                        args.savedir, f'gen_{epoch}_{i}.jpg'))
                # batch checkpoint, cause my gpu sucks
                if epoch > 1 or i != 0:
                    save_checkpoint(
                        gen, gen_optim, config.CHECKPOINT_GEN, epoch, i)
                    save_checkpoint(critic, critic_optim,
                                    config.CHECKPOINT_CRITIC, epoch, i)
                step += 1
        # epoch checkpoint
        save_checkpoint(gen, gen_optim, config.CHECKPOINT_GEN, epoch)
        save_checkpoint(critic, critic_optim, config.CHECKPOINT_CRITIC, epoch)
    # close writers
    writer_real.close()
    writer_fake.close()
    writer_critic.close()


if __name__ == '__main__':
    torch.manual_seed(69)
    parser = argparse.ArgumentParser(
        description='Anime WGAN-GP')
    parser.add_argument('--datadir', type=str, default=None,
                        help='Training data directory')
    parser.add_argument('--glr', type=float, default=1e-4,
                        help='Generator learning rate')
    parser.add_argument('--clr', type=float, default=1e-4,
                        help='Critic learning rate')
    parser.add_argument('--csteps', type=int, default=5,
                        help='Number of critic training steps')
    parser.add_argument('--lambdagp', type=int, default=10,
                        help='Gradient penalty weight')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--savedir', type=str,
                        default='samples', help='Generated samples directory')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume from a checkpoint')
    args = parser.parse_args()
    print('------------ Arguments -------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print('------------------------------------')
    train(args)
