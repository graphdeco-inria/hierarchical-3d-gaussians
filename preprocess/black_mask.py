import os
import re
import cv2
import numpy as np
from joblib import delayed, Parallel
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir')
    args = parser.parse_args()

images_dir = os.path.join(args.project_dir, "camera_calibration/rectified/images")
masks_dir = os.path.join(args.project_dir, "camera_calibration/rectified/masks")

folders = os.listdir(images_dir)
if "jpg" in folders[0]:
    all_img_names = folders
else:
    all_img_names = []
    for folder in folders:
        img_names = os.listdir(f"{images_dir}/{folder}")
        img_names = [f"{folder}/{img_name}" for img_name in img_names]
        all_img_names += img_names

def split_mask(name):
    img = cv2.imread(f"{images_dir}/{name}", cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(f"{masks_dir}/{name[:-4]}.png", cv2.IMREAD_UNCHANGED)
    mask = cv2.dilate(mask, np.ones([5, 5]))
    img[mask == 0] = 0
    cv2.imwrite(f"{images_dir}/{name}", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

Parallel(n_jobs=-1, backend="threading")(
    delayed(split_mask)(name) for name in all_img_names
)
