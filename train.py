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
# from models import Xception

def train(model, train_loader, dev_loader):
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
    tbx = SummaryWriter(f'save/{const.RUN_ID}/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for e in range(const.NUM_EPOCHS):
        print('-' * 50)
        print(f'Epoch {e}')
        losses = AverageMeter()

        end_time = time.time()
        for phase in ('train',): # 'val'
            if phase == 'train':
                model.train()
                dataloader = train_loader
                num_steps = min(len(train_loader), const.MAX_STEPS_PER_EPOCH)
            else:
                model.eval()
                dataloader = dev_loader
                num_steps = min(len(dev_loader), const.MAX_STEPS_PER_EPOCH)

            epoch = const.LAST_SAVE / num_steps + e
            for i, (input_, target) in enumerate(tqdm(dataloader)):
                if i >= num_steps:
                    break
                input_ = input_.to(device)
                target = target.to(device)

                output = model(input_)
                loss = criterion(output, target)
                losses.update(loss.data.item(), input_.size(0))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if i % const.PLT_FREQ == 0:
                    tbx.add_scalar(phase + '/loss', losses.val, epoch*num_steps+i)

                if i % const.LOG_FREQ == 0 and phase == 'train':
                    print(f'{epoch} [{i}/{num_steps}]\t loss {losses.val:.4f} ({losses.avg:.4f})\t')
                    torch.save(model.state_dict(), f'save/{const.RUN_ID}/weights_{int(epoch*num_steps+i)}.pth')
    print(end_time - start_time)

if __name__ == '__main__':
    # train_loader, dev_loader, label_encoder = load_data()
    train_loader = load_data()
    dev_loader = None

    if const.CURR_MODEL == 'xception':
        model = make_model('xception', num_classes=num_classes, pretrained=True, pool=nn.AdaptiveAvgPool2d(1))
        c = 0
        for layer in model.parameters():
            if c < 85:
                layer.requires_grad = False
            else:
                layer.requires_grad = True
            c += 1

    elif const.CURR_MODEL == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, const.NUM_CLASSES)

    if const.RUN_ON_GPU:
        if const.CONTINUE_FROM is not None:
            model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    train(model, train_loader, dev_loader)
