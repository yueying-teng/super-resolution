# -*- coding: utf-8 -*-

import os 
import glob
import numpy as np
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1 import get_default_graph

from EDSR import data_generator as data_generator
from EDSR import config as config
from EDSR import utils as utils
from EDSR import losses as losses
from EDSR import models as models
from EDSR import lr_finder as lr_finder
from EDSR import custom_callbacks as custom_callbacks 

seed = 2020
os.environ['PYTHONHASHSEED'] = '0'                      
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

session_conf = ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = Session(graph=get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


crop_size = config.CROP_SIZE
img_path = config.DATA_PATH
batch_size = config.BATCH_SIZE
scale = config.SCALE
epochs = config.EPOCHS
model_name=config.MODEL_NAME
pretrained_weight = config.PRETRAINED_WEIGHT
finetuned_model_name = config.FINETUNED_MODEL_NAME


def get_data_gen(img_path=img_path, test_size=0.2, batch_size=batch_size, crop_size=crop_size, scale=scale):
    img_path_list = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    img_path_idx = np.arange(len(img_path_list))
    train_img_path, val_img_path, train_img_idx, val_img_idx = train_test_split(img_path_list, img_path_idx, \
                                                                                test_size=test_size, random_state=2020)
    train_gen = data_generator.DataGenerator(img_path_list=train_img_path, crop_size=crop_size, 
                                            batch_size=batch_size, scale=scale, shuffle=True,
                                            crop=True, flip=False, multi_losses=True)

    val_gen = data_generator.DataGenerator(img_path_list=val_img_path, crop_size=crop_size,
                                        batch_size=batch_size, scale=scale, shuffle=True, 
                                        crop=True, flip=False, multi_losses=True)
    
    return train_gen, val_gen, train_img_path, val_img_path


def load_edsr_weight(weight_path, scale=scale, crop_size=crop_size):
    edsr_model = models.edsr(scale=scale, num_filters=64, num_res_blocks=8, \
                            res_block_scaling=None, input_shape=(crop_size, crop_size, 3))
    edsr_model.load_weights(weight_path)

    return edsr_model


def get_output(model, layer_name): 
    """
    get activation maps\content outputs from the pretrained vgg16 model
    """
    return model.get_layer('block{}_conv2'.format(layer_name)).output


def build_EDSR_w_perceptual_loss(edsr_model, crop_size=crop_size):
    img_size = int(crop_size*scale)
    HR = Input((img_size, img_size, 3), name = 'HR')
    LR = Input((crop_size, crop_size, 3), name = 'LR')
    SR = edsr_model(LR)
        
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=Lambda(utils.vgg_preprocess_input)(HR))
    for l in vgg.layers: 
        l.trainable = False

    vgg_content = Model(HR, [get_output(vgg, output) for output in [3, 4, 5]], name = 'VGG16')
    vgg1 = vgg_content(HR) # to extract features of HR
    vgg2 = vgg_content(SR) # to extract features of SR

    # Perceptual Loss
    loss_cont = Lambda(losses.content_loss, name = 'Content')(vgg1+vgg2)
    model = Model([LR, HR], [SR, loss_cont])
    multi_losses = ['edsr', 'Content']
    weights = [1., 1.]

    opt = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[930, 2000], values=[1e-4, 5e-5, 1e-6]))
    model.compile(optimizer=opt, loss = {loss : 'mae' for loss in multi_losses}, 
                loss_weights = weights, metrics={'edsr' : losses.SSIM})
    
    return model


def train(model_name=model_name, pretrained_weight=pretrained_weight, finetuned_model_name=finetuned_model_name):
    train_gen, val_gen, _, _ = get_data_gen()
    # load pretrained edsr model
    edsr_model = load_edsr_weight(weight_path = pretrained_weight)
    model = build_EDSR_w_perceptual_loss(edsr_model=edsr_model)
    # create callbacks
    checkpoint = ModelCheckpoint(filepath = os.path.join('/work/model', '{}.h5'.format(model_name)), 
                                save_best_only=True, save_weights_only=True,
                                monitor = 'val_loss')
    log_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '{}.log'.format(model_name)
    log_path = os.path.join('/work/logs', log_filename) 
    csv_logger = CSVLogger(log_path, separator=',', append=False)

    callbacks_list = [checkpoint, csv_logger]
    model.fit(train_gen, validation_data=val_gen, callbacks=callbacks_list, epochs=epochs)

    # save the weights for the edsr part
    filename = '{}.h5'.format(finetuned_model_name)
    save_dir = os.path.join('/work/model', filename)
    edsr_model.save_weights(save_dir)


def show_one_model_results(weight_path, scale=scale):
    # show HR, LR, SR images from a trained model
    _, _, _, val_img_path = get_data_gen()
    model = load_edsr_weight(weight_path = weight_path, crop_size=None)
    utils.predict_and_show(val_img_path[:5], model=model, scale=scale, multi_255=True)


def compare_two_models_results(weight_path1, weight_path2, scale=scale):
    # comapre the SR predictions from two different models - LR, SR1, SR2 images will be displayed 
    _, _, _, val_img_path = get_data_gen()
    model1 = load_edsr_weight(weight_path = weight_path1, crop_size=None)
    model2 = load_edsr_weight(weight_path = weight_path2, crop_size=None)
    utils.compare_two_models(val_img_path[2:7], model1, model2, scale=scale, SRCNN=False, multi_255=True, show_patch=True)


