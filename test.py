#!/usr/bin/env python
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torchvision

from sklearn.preprocessing import LabelEncoder
from cnn_finetune import make_model
from tqdm import tqdm

import const
from dataset import load_test_data
from models import Xception

def inference(data_loader, model) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    ''' Returns predictions and targets, if any. '''
    model.eval()
    activation = nn.Softmax(dim=1)
    all_predicts, all_confs, all_targets = [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for data in tqdm(data_loader):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None

            input = input_.to(device)
            output = model(input)
            output = activation(output)

            confs, predicts = torch.topk(output, const.NUM_TOP_PREDICTS)
            all_confs.append(confs)
            all_predicts.append(predicts)

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    confs = torch.cat(all_confs)
    targets = torch.cat(all_targets) if len(all_targets) else None
    return predicts, confs, targets


def generate_submission(test_loader, model, label_encoder) -> np.ndarray:
    sample_sub = pd.read_csv('data/sample_submission.csv')

    predicts_gpu, confs_gpu, _ = inference(test_loader, model)
    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()

    labels = [label_encoder.inverse_transform(pred) for pred in predicts]
    print('labels', np.array(labels))
    print('confs', np.array(confs))

    sub = test_loader.dataset.df
    def concat(label: np.ndarray, conf: np.ndarray) -> str:
        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])
    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]

    sample_sub = sample_sub.set_index('id')
    sub = sub.set_index('id')
    sample_sub.update(sub)
    sample_sub.to_csv('submission.csv')


if __name__ == '__main__':
    test_loader = load_test_data()
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy')
    if const.CURR_MODEL == 'xception':
        model = make_model('xception', num_classes=const.NUM_CLASSES)

    elif const.CURR_MODEL == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, const.NUM_CLASSES)

    elif const.CURR_MODEL == 'attention':
        model = Xception(const.NUM_CLASSES)

    if const.RUN_ON_GPU:
        model.load_state_dict(torch.load(const.CONTINUE_FROM))
        model.cuda()

    generate_submission(test_loader, model, label_encoder)
