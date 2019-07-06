import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import const
if const.RUN_ON_GPU:
    sys.path.insert(0, '/home/zephyrnx_gmail_com/aptos-blindness/assets/pretrained-models.pytorch-master')
    sys.path.insert(0, '/home/zephyrnx_gmail_com/aptos-blindness/assets/pytorch-cnn-finetune-master')

import pretrainedmodels
from cnn_finetune import make_model

def compute_size(in_size, kernel_size, stride=1, padding=0):
    return (in_size - kernel_size + 2*padding) // stride + 1

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(84000, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        b = x.shape[0] # (b, 3, 600, 450)
        x = F.relu(self.conv1(x)) # (b, 5, 300, 225)
        x = self.pool(x) # (b, 5, 300, 225)
        x = x.view(b, -1) # (b, 84000)
        x = F.relu(self.fc1(x)) # (b, 84000) to (b, 64)
        x = self.fc2(x) # (b, 1)
        return x.squeeze()

class Xception(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = make_model('xception',
                                   num_classes=1,
                                   pretrained=True,
                                   pool=nn.AdaptiveMaxPool2d(1),
                                   input_size=const.INPUT_SHAPE)
        for layer in self.xception.parameters():
            layer.requires_grad = True

    def forward(self, input):
        x = self.xception(input)
        return x.squeeze()

class ResNet101():
    def __init__(self):
        self.model = pretrainedmodels.__dict__['resnet101'](pretrained=None)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.last_linear = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1),
        )

if __name__ == '__main__':
    x = torch.ones((32, 3, 299, 299))
    model = Xception()
    result = model(x)
    print(result)
