KAGGLE_MODE = False
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pretrainedmodels
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm

if KAGGLE_MODE:
    package_dir = '../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/'
    sys.path.insert(0, package_dir)

ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_SHAPE = (224, 224)
CURR_MODEL = 'mmmodel'
DATA_PATH = '../input/aptos2019-blindness-detection/' if KAGGLE_MODE else 'data/'
TTA = 10
BATCH_SIZE = 32
CUTOFFS = np.array([0.5, 1.5, 2.5, 3.5])

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, mode='test'):
        self.mode = mode
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
        folder = 'train_images/' if self.mode == 'train' else 'test_images'
        img_name = os.path.join(DATA_PATH + folder, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        if self.mode == 'train':
            label = self.data.loc[idx, 'diagnosis']
            return image, label
        return image

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

    model = ResNet101().model
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.load_state_dict(torch.load(f'../input/{CURR_MODEL}/model.bin'))
    model.cuda()

    test_dataset = RetinopathyDataset(csv_file=DATA_PATH + 'sample_submission.csv')
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_preds = np.zeros((TTA, len(test_dataset)))
    for t in range(TTA):
        for i, data in enumerate(tqdm(test_data_loader)):
            pred = model(data.to(device))
            test_preds[t, i*BATCH_SIZE : (i+1)*BATCH_SIZE] = pred.detach().cpu().squeeze().numpy().ravel().reshape(1, -1)

    test_preds = test_preds.sum(axis=0) / float(TTA)
    test_preds = truncated(test_preds)

    sample = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    sample.diagnosis = test_preds.astype(int)
    sample.to_csv('submission.csv', index=False)
