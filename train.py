#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

import const
from dataset import load_data
from util import AverageMeter
from kappa import quadratic_weighted_kappa
from models import Xception, ResNet101, SimpleCNN

if const.USE_LOGGER:
    from tensorboardX import SummaryWriter

def train(model, train_loader, dev_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    if const.USE_LOGGER:
        tbx = SummaryWriter(f'save/{const.RUN_ID}/')

    for e in range(const.NUM_EPOCHS):
        print('-' * 50)
        print(f'Epoch {e}')
        losses = AverageMeter()
        for phase in ('train', 'val'):
            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                    num_steps = min(len(train_loader), const.MAX_STEPS_PER_EPOCH)
                else:
                    model.eval()
                    dataloader = dev_loader
                    num_steps = min(len(dev_loader), const.MAX_STEPS_PER_EPOCH)

                epoch = const.LAST_SAVE / num_steps + e
                for i, (input, target) in enumerate(tqdm(dataloader)):
                    if i >= num_steps:
                        break
                    input = input.to(device)
                    target = target.float().to(device)
                    output = model(input)
                    loss = criterion(output, target)
                    losses.update(loss.data.item(), input.size(0))

                    if phase == 'train':
                        # print(output.detach().cpu().numpy(), target.cpu().numpy(), loss.detach().cpu().numpy())
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                    if i % 100 == 0:
                        if const.USE_LOGGER:
                            quadratic_kappa = quadratic_weighted_kappa(targ, output.detach().cpu().int().numpy())
                            iter = int(epoch*num_steps+i)
                            tbx.add_scalar(phase + '/loss', losses.val, iter)
                            tbx.add_scalar(phase + '/kappa', quadratic_kappa, iter)

                        if i % 5000 == 0 and phase == 'train':
                            quadratic_kappa = quadratic_weighted_kappa(target, output.detach().cpu().int().numpy())
                            print(f'{epoch} [{i}/{num_steps}]\t'
                                  f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                  f'kappa {quadratic_kappa:.4f}')
                            torch.save(model.state_dict(), f'save/{const.RUN_ID}/weights_{iter}.pth')
        # lr_scheduler.step()

if __name__ == '__main__':
    train_loader, dev_loader = load_data()
    model = Xception()

    if const.RUN_ON_GPU:
        if const.CONTINUE_FROM is not None:
            model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    train(model, train_loader, dev_loader)
