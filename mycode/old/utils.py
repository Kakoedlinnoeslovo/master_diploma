import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.layers import Conv2D
from keras.layers import Activation, BatchNormalization, SpatialDropout2D
import keras.backend as K



def get_mask(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, contours, -1, 0, -1)
    # remove the contours from the image and show the resulting images
    img = cv2.bitwise_and(image, image, mask=mask)
    return img


class FolderViewer:
    def __init__(self):
        pass

    def get_folder_list(self, path):
        return os.listdir(path)

    def get_files(self, path, format = 'jpg'):
        onlyfiles = [f for f in listdir(path) if (isfile(join(path, f))) and f.split('.')[-1] == format]
        return onlyfiles

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)


def _run_segmentation(folder_name = "melanoma"):
    viewer = FolderViewer()
    in_mpath = "../data/melanoma/"
    out_mpath_mask = "../data/out/test/melanoma_mask/"
    in_bpath = "../data/benign/"
    out_bpath_mask = "../data/out/test/benign_mask/"

    if folder_name == "melanoma":
        files = viewer.get_files(in_mpath, format='jpg')
        temp_outpath = out_mpath_mask
        temp_inpath = in_mpath

    elif folder_name == "benign":
        files = viewer.get_files(in_bpath, format='jpg')
        temp_outpath = out_bpath_mask
        temp_inpath = in_bpath
    else:
        print("folder_name should be melanoma or benign")
        return

    viewer.create_dir(temp_outpath)
    for i, file in enumerate(tqdm(files)):
        mask = get_mask(temp_inpath + file)
        temp_path = temp_outpath + file
        if i % 100 == 0:
            print('\n' + temp_path)
        cv2.imwrite(temp_path, mask)


def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv

def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unit_test():
    path = "../data/benign/0000.jpg"
    img = get_mask(path)
    plt.imshow(img)
    plt.show()
    cv2.imwrite("../data/test/0000.jpg", img)

if __name__ == "__main__":
    #unit_test()
    _run_segmentation('melanoma')