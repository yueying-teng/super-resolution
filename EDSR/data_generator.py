# -*- coding: utf-8 -*-
import numpy as np
import math
import os
import random
import tensorflow as tf
import glob
import cv2
from random import seed
np.random.seed(2020)
seed(2020)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_path_list, crop_size, batch_size, scale=2, shuffle=True,
                crop=True, keep_dim=False, flip=False, multi_losses=False):
        """
        img_path_list: path that contains the images(original RGB images)
        keep_dim: if True, LR and HR images have the same spatial dimension
                  set to True for Res34Unet and SRCNN
                  False for EDSR
        scale: x2 enhancement
        """
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        # NOTE: images are not resized in this implementation. the valid np array 
        # output relies on using the randon_crop function to keep the dimension
        # consistent 
        self.crop = crop
        self.flip = flip
        self.scale = scale
        self.multi_losses = multi_losses
        self.img_path_list = img_path_list 
        self.keep_dim = keep_dim

    def __len__(self):
        # number of batched per epoch
        return int(math.ceil(len(self.img_path_list) / float(self.batch_size)))

    def on_epoch_end(self):
        # shuffle the indices after each epoch during training
        self.index = range(len(self.img_path_list))
        if self.shuffle == True:
            self.index = random.sample(self.index, len(self.index))
    
    def random_crop(self, lr_img, hr_img):
        lr_crop_size = self.crop_size // self.scale if not self.keep_dim else self.crop_size
        lr_img_shape = lr_img.shape[:2]

        lr_w = np.random.randint(0, lr_img_shape[1] - lr_crop_size + 1)
        lr_h = np.random.randint(0, lr_img_shape[0] - lr_crop_size + 1)
        if self.keep_dim:
            hr_w = lr_w
            hr_h = lr_h
        else:
            hr_w = lr_w * self.scale
            hr_h = lr_h * self.scale

        lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_img_cropped = hr_img[hr_h:hr_h + self.crop_size, hr_w:hr_w + self.crop_size]

        return lr_img_cropped, hr_img_cropped

    def random_flip(self, lr_img, hr_img):
        n = random.randint(0,3)
        flipped_lr_img = np.rot90(lr_img, n)
        flipped_hr_img = np.rot90(hr_img, n)

        return flipped_lr_img, flipped_hr_img

    def __getitem__(self, idx):
        batch_hr_img =[]
        batch_lr_img = []
        batch_img_path = self.img_path_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        for path in batch_img_path:
            hr_img = cv2.imread(path)
            if hr_img is None:
                continue
            h, w, _ = hr_img.shape
            # if any side is not an even number, reduce the hr_img by one row or column
            # so w and h are divisiable in cv2.resize
            if h%2 !=0:
                hr_img = hr_img[:h-1,:,:]
            if w%2 !=0:
                hr_img = hr_img[:, :w-1,:]
            # one resize operation to produce training data and labels for EDSR that does feature extraction on lr images 
            lr_img = cv2.resize(hr_img, (w // self.scale, h // self.scale)) # bilinear interpolation is the default
            if self.keep_dim:
                lr_img = cv2.resize(lr_img, (w, h))
            
            if self.flip:
                lr_img, hr_img = self.random_flip(lr_img, hr_img)
            # so far images are either flipped or not
            if self.crop:
                lr_img, hr_img = self.random_crop(lr_img, hr_img)

            batch_hr_img.append(hr_img)
            batch_lr_img.append(lr_img)

        data, label = np.array(batch_lr_img).astype('float32')/255, np.array(batch_hr_img).astype('float32')/255
        if self.multi_losses:
            return [data, label],  [label, np.zeros((self.batch_size,))]
            # return [data, label],  np.zeros((self.batch_size,))
        else: 
            return data, label



class SRCNNDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_path_list, crop_size, batch_size, scale=2, shuffle=True, crop=True, flip=False):
        """
        img_path_list:
        path that contains the images(original RGB images)
        """
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.crop = crop
        self.flip = flip
        self.scale = scale
        self.img_path_list = img_path_list 
      
    def __len__(self):
        # number of batched per epoch
        return int(math.ceil(len(self.img_path_list) / float(self.batch_size)))

    def on_epoch_end(self):
        # shuffle the indices after each epoch during training
        self.index = range(len(self.img_path_list))
        if self.shuffle == True:
            self.index = random.sample(self.index, len(self.index))
    
    def random_crop(self, lr_img, hr_img):
        lr_img_shape = lr_img.shape[:2]

        lr_w = np.random.randint(0, lr_img_shape[1] - self.crop_size + 1)
        lr_h = np.random.randint(0, lr_img_shape[0] - self.crop_size + 1)

        lr_img_cropped = lr_img[lr_h:lr_h + self.crop_size, lr_w:lr_w + self.crop_size]
        hr_img_cropped = hr_img[lr_h:lr_h + self.crop_size, lr_w:lr_w + self.crop_size]

        return lr_img_cropped, hr_img_cropped

    def __getitem__(self, idx):
        batch_hr_img =[]
        batch_lr_img = []
        batch_img_path = self.img_path_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        for path in batch_img_path:
            hr_img = cv2.imread(path)
            if hr_img is None:
                continue
            h, w, _ = hr_img.shape
            # two resize operations to produce training data and labels for SRCNN 
            lr_img = cv2.resize(hr_img, (w // self.scale, h // self.scale)) # bilinear interpolation is the default
            lr_img = cv2.resize(lr_img, (w, h))
            # TODO: change the crop and flip logic to a similar one as above generator
            if self.crop == False:
                batch_hr_img.append(hr_img)
                batch_lr_img.append(lr_img)
            else:
                cropped_lr_img, cropped_hr_img = self.random_crop(lr_img, hr_img)
                if self.flip:
                    n = random.randint(0,3)
                    flipped_hr_img = np.rot90(cropped_hr_img, n)
                    flipped_lr_img = np.rot90(cropped_lr_img, n)
                    batch_hr_img.append(flipped_hr_img)
                    batch_lr_img.append(flipped_lr_img)

                else:
                    batch_hr_img.append(cropped_hr_img)
                    batch_lr_img.append(cropped_lr_img)

        return np.array(batch_lr_img).astype('float32')/255, np.array(batch_hr_img).astype('float32')/255