# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf


def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None, input_shape=(None,None,3)):
    """
    Creates an EDSR model.
    """
    x_in = Input(input_shape)
    x = Lambda(normalize)(x_in)
    x = b = Conv2D(num_filters, 3, padding='same')(x)
    
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
    x = Lambda(denormalize)(x)

    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    """
    Creates an EDSR residual block.
    """
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])

    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """
        Sub-pixel convolution.
        """
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def normalize(x, DIV2K_RGB_MEAN=[0,0,0]):
    # return (x - DIV2K_RGB_MEAN) / 127.5
    return x

def denormalize(x, DIV2K_RGB_MEAN=[0,0,0]):
    # return x * 127.5 + DIV2K_RGB_MEAN
    return x