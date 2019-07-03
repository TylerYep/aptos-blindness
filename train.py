#!/usr/bin/env python
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tensorboardX import SummaryWriter
from cnn_finetune import make_model
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
    tbx = SummaryWriter(f'save/{const.RUN_ID}/')
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
                input = input.to(device)
                target = targ.float().reshape(-1, 1).to(device)
                output = model(input)
                loss = criterion(output, target)
                losses.update(loss.data.item(), input.size(0))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if i % 100 == 0:
                    iter = int(epoch*num_steps+i)
                    quadratic_kappa = quadratic_weighted_kappa(targ, output.detach().int().numpy())
                    tbx.add_scalar(phase + '/loss', losses.val, iter)
                    tbx.add_scalar(phase + '/kappa', quadratic_kappa, iter)

                    if i % 5000 == 0 and phase == 'train':
                        print(f'{epoch} [{i}/{num_steps}]\t loss {losses.val:.4f} ({losses.avg:.4f})\t kappa {quadratic_kappa:.4f}')
                        torch.save(model.state_dict(), f'save/{const.RUN_ID}/weights_{iter}.pth')
    print(end_time - start_time)

if __name__ == '__main__':
    train_loader, dev_loader = load_data()

    if const.CURR_MODEL == 'xception':
        model = Xception()

    if const.RUN_ON_GPU:
        if const.CONTINUE_FROM is not None:
            model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    train(model, train_loader, dev_loader)
