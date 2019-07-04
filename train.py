#!/usr/bin/env python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tensorboardX import SummaryWriter
from tqdm import tqdm

import const
from dataset import RetinopathyDataset, load_data
from util import AverageMeter
from kappa import quadratic_weighted_kappa
from models import Xception

def train(model, train_loader, dev_loader):
    start_time = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
    if const.RUN_ON_GPU: tbx = SummaryWriter(f'save/{const.RUN_ID}/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for e in range(const.NUM_EPOCHS):
        print('-' * 50)
        print(f'Epoch {e}')
        losses = AverageMeter()

        end_time = time.time()
        for phase in ('train', 'val'):
            if phase == 'train':
                model.train()
                dataloader = train_loader
                num_steps = min(len(train_loader), const.MAX_STEPS_PER_EPOCH)
            else:
                model.eval()
                dataloader = dev_loader
                num_steps = min(len(dev_loader), const.MAX_STEPS_PER_EPOCH)

            epoch = const.LAST_SAVE / num_steps + e
            for i, (input, targ) in enumerate(tqdm(dataloader)):
                if i >= num_steps:
                    break
                visualize(input, targ)
                break

                input = input.to(device)
                target = targ.float().to(device)
                output = model(input).squeeze()
                loss = criterion(output, target)
                losses.update(loss.data.item(), input.size(0))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if i % 100 == 0 and const.RUN_ON_GPU:
                    iter = int(epoch*num_steps+i)
                    quadratic_kappa = quadratic_weighted_kappa(targ, output.detach().cpu().int().numpy())
                    tbx.add_scalar(phase + '/loss', losses.val, iter)
                    tbx.add_scalar(phase + '/kappa', quadratic_kappa, iter)

                    if i % 5000 == 0 and phase == 'train':
                        print(f'{epoch} [{i}/{num_steps}]\t loss {losses.val:.4f} ({losses.avg:.4f})\t kappa {quadratic_kappa:.4f}')
                        torch.save(model.state_dict(), f'save/{const.RUN_ID}/weights_{iter}.pth')
    print(end_time - start_time)

def visualize(input, target):
    fig = plt.figure(figsize=(25, 16))
    for i in range(input.shape[0]):
        ax = fig.add_subplot(2, 2, i+1)
        plt.imshow(input[i].permute(1, 2, 0))
        ax.set_title(int(target[i]))
    plt.show()

if __name__ == '__main__':
    train_loader, dev_loader = load_data()

    model = Xception()

    if const.RUN_ON_GPU:
        if const.CONTINUE_FROM is not None:
            model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    train(model, train_loader, dev_loader)
