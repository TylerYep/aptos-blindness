KAGGLE_MODE = True
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
    sys.path.append('../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/')

from efficientnet_pytorch import EfficientNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
INPUT_SHAPE = (256, 256)
DATA_PATH = '../input/aptos2019-blindness-detection/' if KAGGLE_MODE else 'data/'
SAVE_PATH = '../input/' + RUN_ID if KAGGLE_MODE else 'save/'
CURR_MODEL_WEIGHTS = SAVE_PATH + CURR_WEIGHTS
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
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        def crop_image_from_gray(img, tol=7):
            if img.ndim == 2:
                mask = img > tol
                return img[np.ix_(mask.any(1),mask.any(0))]
            elif img.ndim == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                mask = gray_img > tol

                check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
                if (check_shape != 0): # image is too dark so that we crop out everything,
                    img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                    img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                    img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
                    img = np.stack([img1, img2, img3], axis=-1)
                return img

        img_name = os.path.join(DATA_PATH + 'test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, INPUT_SHAPE)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 30), -4, 128)
        image = transforms.ToPILImage()(image)
        image = self.transform(image)
        return image

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
    try:
        sample = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
    except:
        sample = pd.read_csv('../input/sample_submission.csv')

    if len(sample) < 2000:
        sample.to_csv('submission.csv',index=False)
        return

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = EfficientNet.from_name('efficientnet-b0')
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, 1)
        model.eval()

        if KAGGLE_MODE:
#             weights = torch.load(CURR_MODEL_WEIGHTS)
#             model.load_state_dict(weights, strict=False)
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
        sample.diagnosis = test_preds.astype(int)
        sample.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    test()