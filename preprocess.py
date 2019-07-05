#!/usr/bin/env python
import matplotlib.pyplot as plt
import const
from dataset import load_data
import cv2
import glob
import numpy as np

def main():
    train_loader, dev_loader = load_data()
    for i, (input, targ) in enumerate(train_loader):
        target = targ.float()
        visualize(input, target)

def visualize(input, target):
    fig = plt.figure(figsize=(25, 16))
    for i in range(input.shape[0]):
        ax = fig.add_subplot(2, 2, i+1)
        plt.imshow(input[i].permute(1, 2, 0))
        ax.set_title(int(target[i]))
    plt.show()

def scaleRadius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(axis=1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0,0), fx=s, fy=s)

def preprocess(mode='train'):
    scale = 300
    for f in glob.glob(f'data/{mode}_images/*.png'):
        orig = cv2.imread(f)
        # Scale image to a given radius.
        a = scaleRadius(orig, scale)
        # Subtract local mean color.
        a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
        # Remove outer 10%.
        b = np.zeros(a.shape)
        cv2.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale*0.9), (1, 1, 1), -1, 8, 0)
        a = a*b + 128*(1-b)
        filename = f[f.rfind('/')+1:]
        cv2.imwrite(f'data/preprocessed_{mode}/{filename}', a)
        break

if __name__ == '__main__':
    preprocess()

