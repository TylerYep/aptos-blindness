#!/usr/bin/env python
import cv2
import glob
import numpy as np
from tqdm import tqdm

def scale_radius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(axis=1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0,0), fx=s, fy=s)

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape != 0): # image is too dark so we cropped out everything,
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def circle_crop(img):
    height, width, depth = img.shape
    x = width // 2
    y = height // 2
    r = np.amin((x,y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img, r

def preprocess(mode='train'):
    scale = 300
    for f in tqdm(glob.glob(f'data/{mode}_images/*.png')):
        img = cv2.imread(f)

        # Scale image to a given radius.
        img, r = circle_crop(img)

        # Subtract local mean color.
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), scale/30), -4, 128)

        # Remove outer 5%.
        b = np.zeros(img.shape)
        cv2.circle(b, (img.shape[1]//2, img.shape[0]//2), int(r*0.95), (1, 1, 1), -1, 8, 0)
        img = img*b + 128*(1-b)

        filename = f[f.rfind('/')+1:]
        cv2.imwrite(f'data/preprocessed_{mode}/{filename}', img)

if __name__ == '__main__':
    preprocess('train')
    preprocess('test')
