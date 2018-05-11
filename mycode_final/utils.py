import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt



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


def unit_test():
    path = "../data/benign/0000.jpg"
    img = get_mask(path)
    plt.imshow(img)
    plt.show()
    cv2.imwrite("../data/test/0000.jpg", img)

if __name__ == "__main__":
    unit_test()