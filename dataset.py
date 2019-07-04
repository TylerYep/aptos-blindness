import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import const

class RetinopathyDataset(Dataset):
    def __init__(self, csv_file, mode='train'):
        self.mode = mode
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize(const.INPUT_SHAPE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder = 'train_images/' if self.mode == 'train' else 'test_images'
        img_name = os.path.join(const.DATA_PATH + folder, self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        if self.mode == 'train':
            label = self.data.loc[idx, 'diagnosis']
            return image, label
        return image


def load_data():
    train_dataset = RetinopathyDataset(const.TRAIN_CSV)
    dev_dataset = RetinopathyDataset(const.DEV_CSV)
    train_data_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    dev_data_loader = DataLoader(dev_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    return train_data_loader, dev_data_loader



if __name__ == '__main__':
    train_dataset = RetinopathyDataset(const.TRAIN_CSV)
    # test_dataset = RetinopathyDataset(const.DATA_PATH + 'sample_submission.csv', 'test')
    print(train_dataset[5])
