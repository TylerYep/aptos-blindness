KAGGLE_MODE = True
GPU_MODE = False
RUN_ID = 'xceptioncolab/'
CURR_WEIGHTS = 'weights_4655.pth'

import sys
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

if KAGGLE_MODE:
    sys.path.insert(0, '../input/pretrainedmodels/pretrainedmodels/pretrained-models.pytorch-master/')
    sys.path.insert(0, '../input/cnnfinetune/pytorch-cnn-finetune-master/pytorch-cnn-finetune-master/')
elif GPU_MODE:
    sys.path.insert(0, '/home/zephyrnx_gmail_com/aptos-blindness/assets/pretrained-models.pytorch-master')
    sys.path.insert(0, '/home/zephyrnx_gmail_com/aptos-blindness/assets/pytorch-cnn-finetune-master')

import pretrainedmodels
from cnn_finetune import make_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_SHAPE = (600, 450)
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
        self.preprocess()
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SHAPE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(DATA_PATH + 'test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return image

    def __getitem__(self, idx):
        def scaleRadius(img, scale):
            x = img[img.shape[0]//2, :, :].sum(axis=1)
            r = (x > x.mean() / 10).sum() / 2
            s = scale * 1.0 / r
            return cv2.resize(img, (0,0), fx=s, fy=s)
        scale = 300
        img_name = os.path.join(DATA_PATH + 'test_images', self.data.loc[idx, 'id_code'] + '.png')
        orig = cv2.imread(img_name)
        # Scale image to a given radius.
        a = scaleRadius(orig, scale)
        # Subtract local mean color.
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
        # Remove outer 10%.
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
        img = a*b + 128*(1-b)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = Image.open(img_name)
        image = self.transform(img)
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

def test():
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Xception()
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

if __name__ == '__main__':
    test()
