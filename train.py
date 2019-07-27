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
from apex import amp

if const.RUN_ON_GPU:
    import sys
    sys.path.append('/content/aptos-blindness/assets')
else:
    import viz

from efficientnet_pytorch import EfficientNet

if const.USE_TENSORBOARD:
    from tensorboardX import SummaryWriter

def train(model):
    train_loader, dev_loader = load_data()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE, weight_decay=1e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    if const.USE_TENSORBOARD:
        tbx = SummaryWriter(f'save/{const.RUN_ID}/')

    best_avg_loss = 100.0
    for e in range(const.NUM_EPOCHS):
        print('-' * 50)
        print(f'Epoch {e}')
        losses = AverageMeter()
        for phase in ('train', 'val'):
            running_loss = 0.0
            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    model.train()
                    dataloader = tqdm(train_loader)
                    num_steps = min(len(train_loader), const.MAX_STEPS_PER_EPOCH)
                else:
                    model.eval()
                    dataloader = tqdm(dev_loader)
                    num_steps = min(len(dev_loader), const.MAX_STEPS_PER_EPOCH)

                epoch = const.LAST_SAVE / num_steps + e

                for i, (input, target) in enumerate(dataloader):
                    # if i >= num_steps:
                    #     break
                    # viz.visualize(input, target)
                    input = input.to(device)
                    target = target.float().to(device)
                    output = model(input).squeeze()
                    loss = criterion(output, target)
                    losses.update(loss.data.item(), input.size(0))
                    running_loss += loss.item() * input.size(0)
                    dataloader.set_postfix(loss=losses.avg)

                    if phase == 'train':
                        # print(output.detach().cpu().numpy(), target.cpu().numpy(), loss.detach().cpu().numpy())
                        optimizer.zero_grad()
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        optimizer.step()

                    if const.USE_TENSORBOARD:
                        iter = int(epoch*num_steps+i)
                        quadratic_kappa = quadratic_weighted_kappa(target.cpu().numpy(), output.detach().cpu().int().numpy())
                        tbx.add_scalar(phase + '/loss', losses.val, iter)
                        tbx.add_scalar(phase + '/kappa', quadratic_kappa, iter)

                    if i % 500 == 0 and phase == 'train':
                        quadratic_kappa = quadratic_weighted_kappa(target.cpu().numpy(), output.detach().cpu().int().numpy())
                        print(f'{epoch} [{i}/{num_steps}]\t'
                              f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                              f'kappa {quadratic_kappa:.4f}')

                if phase == 'val':
                    print(f'DEV loss {losses.val:.4f} ({losses.avg:.4f})\t kappa {quadratic_kappa:.4f}')

            epoch_loss = running_loss / len(dataloader)
            print(f'Epoch Loss: {epoch_loss:.4f}')
            if epoch_loss < best_avg_loss:
                best_avg_loss = epoch_loss
                torch.save(model.state_dict(), f'save/{const.RUN_ID}_weights_{e}.pth')
        scheduler.step()

if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b1')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 1)

    if const.RUN_ON_GPU:
        if const.CONTINUE_FROM is not None:
            model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    train(model)
