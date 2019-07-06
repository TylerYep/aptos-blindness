#!/usr/bin/env python
import cv2
import glob
import numpy as np
from tqdm import tqdm

def preprocess(mode='train'):
    def scaleRadius(img, scale):
        x = img[img.shape[0]//2, :, :].sum(axis=1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        return cv2.resize(img, (0,0), fx=s, fy=s)

    scale = 300
    for f in tqdm(glob.glob(f'data/{mode}_images/*.png')):
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

if __name__ == '__main__':
    preprocess('train')
    preprocess('test')

