# -*- coding: utf-8 -*-
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras import backend as K

    
def vgg_preprocess_input(x):
    # BGR Mean values
    vgg_mean = np.array([103.939, 116.779, 123.68], dtype='float32').reshape((1,1,3))
    return x - vgg_mean/255.


def get_PSNR(truth, pred):
    """
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    assumes RGB image arrays in range [0, 255]
    """
    truth_data = np.array(truth, dtype=float)
    pred_data = np.array(pred, dtype=float)
    diff = pred_data - truth_data
    diff = diff.flatten()
    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255.) - 10*math.log10(rmse)


def get_SSIM(truth, pred): 
    """
    for a single image
    """
    return ssim(truth, pred, multichannel=True)


def show_batch_imgs(idx, demo_ori_img, demo_lr_img, num_col=2, num_row=1, figsize = (7, 3)):
    plt.figure(figsize=figsize) 
    truth = demo_ori_img[idx]
    lr = demo_lr_img[idx]
    images = [truth, lr]
    
    for i, img in enumerate(images):
        plt.subplot(num_row, num_col, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def show_pred(idx, demo_ori_img, demo_lr_img, demo_pred_img, \
              title1='ground truth', title2='LR(SSIM)', title3='SR(SSIM)', num_col=3, num_row=1, figsize = (30, 10)):
    plt.figure(figsize=figsize) 
    truth = demo_ori_img[idx]
    lr = demo_lr_img[idx]
    sr = demo_pred_img[idx]
    
    images = [truth, lr, sr]
    # increase the size of lr again as both PSNR and SSIM requires images to be of the same dimension
    h, w, _ = truth.shape
    lr = cv2.resize(lr, (w, h))
    lr_ssim = get_SSIM(truth, lr)
    pred_ssim = get_SSIM(truth, sr)
    titles = [title1, 
             title2 + ' '+ str(round(lr_ssim,3)), 
             title3 + ' '+ str(round(pred_ssim,3))]
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(num_row, num_col, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=20)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def predict_one(img_path, model, scale, SRCNN=False, multi_255=False):
    ori_img = cv2.imread(img_path)
    h, w, _ = ori_img.shape

    if not SRCNN:
        # if any side is not an even number, reduce the hr_img by one row or column
        # so w and h are divisiable in cv2.resize
        if h%2 !=0:
            ori_img = ori_img[:h-1,:,:]
        if w%2 !=0:
            ori_img = ori_img[:, :w-1,:]
        lr_img = cv2.resize(ori_img, (w// scale, h// scale))
    else:
        lr_img = cv2.resize(ori_img, (w// scale, h// scale))
        lr_img = cv2.resize(lr_img, (w, h))
    
    if not multi_255:
        model_input = np.expand_dims(lr_img, 0)
        pred_img = model.predict(model_input)
    else:
        model_input = np.expand_dims(lr_img/255, 0)
        pred_img = model.predict(model_input)* 255.

    pred_img[pred_img[:] > 255] = 255
    pred_img[pred_img[:] < 0] = 0
    pred_img = pred_img.astype(np.uint8)

    return ori_img, lr_img, pred_img[0]


def predict_and_show(img_path_list, model, scale, SRCNN=False, multi_255=False):
    demo_ori_img, demo_lr_img, demo_pred_img = [], [], []
    for path in img_path_list:
        ori_img, lr_img, pred_img = predict_one(path, model, scale, SRCNN=SRCNN, multi_255=multi_255) 
        demo_ori_img.append(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
        demo_lr_img.append(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)) 
        demo_pred_img.append(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))

    for i in range(len(img_path_list)):
        show_pred(idx=i, demo_ori_img=demo_ori_img, demo_lr_img=demo_lr_img, demo_pred_img=demo_pred_img)


def compare_two_models(img_path_list, model1, model2, scale, SRCNN=False, multi_255=False, show_patch=True, patch_size=200):
    """
    e.g. model1 could be an EDSR trained on pixe loss
         model2 could be a fine tuned EDSR using perceptual loss
    """
    demo_lr_img, demo_pred_img1, demo_pred_img2 = [], [], []
    for path in img_path_list:
        ori_img, lr_img, pred_img1 = predict_one(path, model1, scale, SRCNN=SRCNN, multi_255=multi_255) 
        _, _, pred_img2 = predict_one(path, model2, scale, SRCNN=SRCNN, multi_255=multi_255) 
        if show_patch:
            h, w, _ = ori_img.shape
            lr_img = cv2.resize(lr_img, (w, h))
            lr_img_shape = lr_img.shape[:2]

            lr_w = np.random.randint(0, lr_img_shape[1] - patch_size + 1)
            lr_h = np.random.randint(0, lr_img_shape[0] - patch_size + 1)

            lr_img = lr_img[lr_h:lr_h + patch_size, lr_w:lr_w + patch_size]
            pred_img1 = pred_img1[lr_h:lr_h + patch_size, lr_w:lr_w + patch_size]
            pred_img2 = pred_img2[lr_h:lr_h + patch_size, lr_w:lr_w + patch_size]
        
        demo_lr_img.append(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB))
        demo_pred_img1.append(cv2.cvtColor(pred_img1, cv2.COLOR_BGR2RGB)) 
        demo_pred_img2.append(cv2.cvtColor(pred_img2, cv2.COLOR_BGR2RGB))

    for i in range(len(img_path_list)):
        show_pred(idx=i, demo_ori_img=demo_lr_img, demo_lr_img=demo_pred_img1, demo_pred_img=demo_pred_img2,\
                  title1='LR', title2='EDSR pixel loss(SSIM)', title3='EDSR perceptual loss(SSIM)',)
