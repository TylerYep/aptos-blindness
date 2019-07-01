import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_finetune import make_model
import const

class Xception(nn.Module):
    def __init__(self, num_classes=const.NUM_CLASSES):
        super().__init__()

        self.xception = make_model('xception', num_classes=num_classes, pretrained=True,
                                   pool=nn.AdaptiveMaxPool2d(1))
        c = 0
        for layer in self.xception.parameters():
            if c < 85:
                layer.requires_grad = False
            else:
                layer.requires_grad = True
            c += 1

    def forward(self, input):
        b, h, w, c = input.shape    # in = (b, 3, 299, 299)
        x = self.xception(input)    # out= (b, 10, 10, 2048)
        return x

if __name__ == '__main__':
    x = torch.ones((32, 3, 299, 299))
    model = Xception()
    result = model(x)
    print(result)
