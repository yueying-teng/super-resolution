# -*- coding: utf-8 -*-
from tensorflow.keras import backend as K
import tensorflow as tf

def PSNR(y_true, y_pred):
    """
    used as a metric to measure simliarity between images 
    minimizing MSE leads to the maximization of this metric

    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def SSIM(y_true, y_pred):
    """
    used as a metric to measure simliarity between images 
    the higher the more similar
    """
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def RMSE(diff): 
    """
    MSE for feature reconstruction, i.e. perceptual loss 
    dim: (1, batch_size)
    """
    return K.expand_dims(K.sqrt(K.mean(K.square(diff), [1,2,3])), 0)
 

content_w=[0.1, 0.8, 0.1]
def content_loss(x):
    """
    perceptual loss
    x is an array of VGG1(HR) outputs appended with an array of VGG2(SR) outputs
    compares x[0] and x[3] 
             x[1] and x[4] and so on
        x[0], x[1], x[2] are features extracted from a HR image using the last three CNN blocks of pretrained vgg16
        x[3], x[4], x[5] are features extracted from a SR image using the last three CNN blocks of pretrained vgg16
    """
    content_loss = 0
    n = len(content_w)
    for i in range(n): 
        content_loss += RMSE(x[i] - x[i+n]) * content_w[i]

    return content_loss