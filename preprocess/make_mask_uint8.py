import os
import re
import cv2
import numpy as np
from joblib import delayed, Parallel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir')
    parser.add_argument('--out_dir')
    args = parser.parse_args()

in_dir = args.in_dir
masks_dir = args.out_dir

folders = os.listdir(in_dir)
if "png" in folders[0]:
    all_img_names = folders
else:
    all_img_names = []
    for folder in folders:
        img_names = os.listdir(f"{in_dir}/{folder}")
        img_names = [f"{folder}/{img_name}" for img_name in img_names]
        all_img_names += img_names

def split_mask(name):
    img = cv2.imread(f"{in_dir}/{name}", cv2.IMREAD_UNCHANGED)
    if img is not None:
        os.makedirs(os.path.dirname(f"{masks_dir}/{name}"), exist_ok=True)
        mask = (img[..., -1] > 250).astype(np.uint8) * 255
        cv2.imwrite(f"{masks_dir}/{name}", (cv2.erode(mask, np.ones([3, 3])) > 250).astype(np.uint8) * 255)

Parallel(n_jobs=-1, backend="threading")(
    delayed(split_mask)(name) for name in all_img_names
)
