import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
import pretrainedmodels
import const

class CustomClassifier(nn.Module):
    def __init__(self, in_feats, num_classes):
        super().__init__()
        self.final_layers = nn.Sequential(
            nn.Linear(in_features=in_feats, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        return self.final_layers(x)

class Xception(nn.Module):
    def __init__(self):
        super().__init__()

        self.xception = make_model('xception', num_classes=1, pretrained=True,
                                   pool=nn.AdaptiveMaxPool2d(1))
        c = 0
        for layer in self.xception.parameters():
            layer.requires_grad = (c >= 85)
            c += 1

    def forward(self, input): # in = (b, 3, 299, 299)
        x = self.xception(input)    # out = (b, 10, 10, 2048) # out = (b, 1, 1, 1)
        return x

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
