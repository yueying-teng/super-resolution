# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import numpy as np
from SRCNN import config

import importlib
importlib.reload(config)

DATA_PATH = config.DATA_PATH
num_random_crops = config.num_random_crops
patch_size = config.patch_size
label_size = config.label_size
scale = config.scale
num_train_img = config.num_train_img


# to load training data patches
def prepare_data(path = DATA_PATH):
    names = sorted(os.listdir(path))
    # use the first 300 images in the raw data path to create training data and label
    nums = num_train_img
    data = np.zeros((nums * num_random_crops, 1, patch_size, patch_size), dtype=np.double)
    label = np.zeros((nums * num_random_crops, 1, label_size, label_size), dtype=np.double)

    for i in range(nums):     
        if 'jpg' in names[i]:
            name = os.path.join(path, names[i])
            print(name)
            hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
            if hr_img is None:
                continue
            shape = hr_img.shape

            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
            hr_img = hr_img[:, :, 0]

            # two resize operation to produce training data and labels
            lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale)) # bilinear interpolation is the default
            lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

            # produce num_random_crops random coordinate to crop training img
            points_x = np.random.randint(0, min(shape[0], shape[1]) - patch_size, num_random_crops)
            points_y = np.random.randint(0, min(shape[0], shape[1]) - patch_size, num_random_crops)

            for j in range(num_random_crops):
                lr_patch = lr_img[points_x[j]: points_x[j] + patch_size, points_y[j]: points_y[j] + patch_size]
                hr_patch = hr_img[points_x[j]: points_x[j] + patch_size, points_y[j]: points_y[j] + patch_size]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                data[i * num_random_crops + j, 0, :, :] = lr_patch
                label[i * num_random_crops + j, 0, :, :] = hr_patch
              
    return data, label

 

def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(np.float32)
    y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        train_data = np.transpose(data, (0, 2, 3, 1))
        train_label = np.transpose(label, (0, 2, 3, 1))
        return train_data, train_label

 