from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from SRCNN import data_prep, config
import cv2
import os 
import matplotlib.pyplot as plt
import numpy as np
import math


DATA_PATH = config.DATA_PATH
num_demo_img = config.num_demo_img 
batch_size = config.batch_size
epochs = config.epochs


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def model():
    SRCNN = Sequential()
    
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='same', use_bias=True))
    # define optimizer
    adam = Adam(lr=0.001)
    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error',  metrics=[PSNRLoss])
    
    return SRCNN


def predict_model():
    SRCNN = Sequential()
    
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True, input_shape=(None, None, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='same', use_bias=True))

    return SRCNN


def train():
    srcnn_model = model()
    print(srcnn_model.summary())
    data, label = data_prep.read_training_data("./training_data.h5")
    checkpoint = ModelCheckpoint("SRCNN.h5", monitor='val_PSNRLoss', save_best_only=True,
                                 save_weights_only=False, mode='max')
    callbacks_list = [checkpoint]

    srcnn_model.fit(data, label, batch_size=batch_size, 
                    validation_split = 0.2,
                    callbacks=callbacks_list, shuffle=True, epochs=epochs)


def predict_one(img_file_name, srcnn_model):
    # INPUT_NAME = 'input_' + img_file_name 
    # OUTPUT_NAME = 'pred_' + img_file_name 
    img_path = os.path.join(DATA_PATH, img_file_name)
    ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    lr_img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite(INPUT_NAME, lr_img)

    Y = np.zeros((1, lr_img.shape[0], lr_img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pred_img = srcnn_model.predict(Y) * 255.
    pred_img[pred_img[:] > 255] = 255
    pred_img[pred_img[:] < 0] = 0
    pred_img = pred_img.astype(np.uint8)
    lr_img_tmp = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)
    lr_img_tmp[:, :, 0] = pred_img[0, :, :, 0]
    pred_img = cv2.cvtColor(lr_img_tmp, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite(OUTPUT_NAME, pred_img)

    return ori_img, lr_img, pred_img


def get_PSNR(truth, pred):
    """
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    """
    # assume RGB image
    truth_data = np.array(truth, dtype=float)
    pred_data = np.array(pred, dtype=float)

    diff = pred_data - truth_data
    diff = diff.flatten()

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255.) - 10*math.log10(rmse)


def show_demo(idx, demo_ori_img, demo_lr_img, demo_pred_img, num_col=3, num_row=1, figsize = (30, 10)):
    plt.figure(figsize=figsize)
    
    truth = demo_ori_img[idx]
    lr = demo_lr_img[idx]
    sr = demo_pred_img[idx]
    images = [truth, lr, sr]
    pred_psnr = get_PSNR(truth, sr)
    lr_psnr = get_PSNR(truth, lr)
    titles = ['ground truth','LR inpnut {}'.format(round(lr_psnr,3)),'SR prediction {}'.format(round(pred_psnr,3))]

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(num_row, num_col, i+1)
        plt.imshow(img)
        plt.title(title, fontsize=20)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()

    
def predict_and_show():
    srcnn_model = predict_model()
    srcnn_model.load_weights("SRCNN.h5")
    # use the last ten images from the raw data to show to resultd of SRCNN 
    names = sorted(os.listdir(DATA_PATH))[-num_demo_img: ]
    
    demo_ori_img, demo_lr_img, demo_pred_img = [], [], []
    for img_file_name in names:
        ori_img, lr_img, pred_img = predict_one(img_file_name, srcnn_model) 
        demo_ori_img.append(cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB))
        demo_lr_img.append(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)) 
        demo_pred_img.append(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))

    for i in range(num_demo_img):
        show_demo(idx=i, demo_ori_img=demo_ori_img, demo_lr_img=demo_lr_img, demo_pred_img=demo_pred_img)

