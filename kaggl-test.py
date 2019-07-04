KAGGLE_MODE = False
GPU_MODE = True
RUN_ID = 'xception1/'
CURR_WEIGHTS = 'weights_552.pth'
import sys
if KAGGLE_MODE:
    sys.path.insert(0, '../input/pretrainedmodels/pretrainedmodels/pretrained-models.pytorch-master/')
    sys.path.insert(0, '../input/cnnfinetune/pytorch-cnn-finetune-master/pytorch-cnn-finetune-master/')
else:
    sys.path.insert(0, 'assets/pretrained-models.pytorch-master/')
    sys.path.insert(0, 'assets/pytorch-cnn-finetune-master/')
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from cnn_finetune import make_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
import pretrainedmodels
import cnn_finetune

ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_SHAPE = (229, 229)
SAVE_PATH = '../input/' if KAGGLE_MODE else 'save/'
if KAGGLE_MODE or GPU_MODE: SAVE_PATH += RUN_ID
CURR_MODEL_WEIGHTS = SAVE_PATH + CURR_WEIGHTS
DATA_PATH = '../input/aptos2019-blindness-detection/' if KAGGLE_MODE else 'data/'
TTA = 10
BATCH_SIZE = 32
CUTOFFS = np.array([0.5, 1.5, 2.5, 3.5])

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SHAPE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder = 'test_images'
        img_name = os.path.join(DATA_PATH + folder, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return image

class Xception(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = make_model('xception', num_classes=1, pretrained=False, pool=nn.AdaptiveMaxPool2d(1))

    def forward(self, input): # in = (b, 3, 299, 299)
        x = self.xception(input)  # out = (b, 1)
        return x

def truncated(test_preds):
    for i, pred in enumerate(test_preds):
        if pred < CUTOFFS[0]:
            test_preds[i] = 0
        elif pred >= CUTOFFS[0] and pred < CUTOFFS[1]:
            test_preds[i] = 1
        elif pred >= CUTOFFS[1] and pred < CUTOFFS[2]:
            test_preds[i] = 2
        elif pred >= CUTOFFS[2] and pred < CUTOFFS[3]:
            test_preds[i] = 3
        else:
            test_preds[i] = 4
        return test_preds

def rounded(test_preds):
    x = np.subtract.outer(test_preds, CUTOFFS)
    y = np.argmin(x, axis=1)
    test_preds = CUTOFFS[y]
    return test_preds

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Xception()
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    if KAGGLE_MODE or GPU_MODE:
        weights = torch.load(CURR_MODEL_WEIGHTS)
        model.load_state_dict(weights, strict=False)
        model.cuda()
    else:
        weights = torch.load(CURR_MODEL_WEIGHTS, map_location=lambda storage, loc: storage)
        model.load_state_dict(weights, strict=False)

    test_dataset = RetinopathyDataset(csv_file=DATA_PATH + 'sample_submission.csv')
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_preds = np.zeros((TTA, len(test_dataset)))
    for t in range(TTA):
        for i, input in enumerate(tqdm(test_data_loader)):
            input = input.to(device)
            pred = model(input)
            test_preds[t, i*BATCH_SIZE : (i+1)*BATCH_SIZE] = pred.detach().cpu().squeeze().numpy().ravel().reshape(1, -1)

    test_preds = test_preds.sum(axis=0) / float(TTA)
    test_preds = truncated(test_preds)

    sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    sample.diagnosis = test_preds.astype(int)
    sample.to_csv('submission.csv', index=False)