from imgaug import augmenters as iaa
import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import io

from mycode import OUTPUT_SIZE, OUTPUT_CHANNELS
from mycode import FolderViewer

class DataReader:
    def __init__(self, data_path, out_path):
        self.data_path = data_path
        self.out_path = out_path
        self.viewer = FolderViewer()

    def read(self, sub_path = "benign/", batch_size = None, is_shuffle = True, is_masked = True):
        temp_path  = self.data_path + sub_path
        imgs_list = self.viewer.get_files(path = temp_path)

        if is_shuffle:
            shuffle(imgs_list)
        if batch_size is None:
            batch_size = len(imgs_list)

        result = np.zeros((batch_size, OUTPUT_SIZE, OUTPUT_SIZE, OUTPUT_CHANNELS), dtype= np.uint8)

        for i, img_path in tqdm(enumerate(imgs_list[:batch_size])):
            if is_masked:
                img = io.imread(img_path)
                img = self.make_masked(img)
                img = self.resize(img, OUTPUT_SIZE)
            else:
                img = io.imread(img_path)
                img = self.resize(img, OUTPUT_SIZE)

            result[i] = img

        return result



    @staticmethod
    def make_masked(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 127, 127, 0)
        img_after, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        cv2.drawContours(mask, contours, -1, 0, -1)
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)
        result_img = mask & img
        return result_img


    def make_augmentations(self):
        pass


    @staticmethod
    def resize(img, output_size):
        dst = cv2.resize(img, (output_size, output_size), interpolation = cv2.INTER_CUBIC)
        return dst


    def _save(self):
        pass


    def _load(self):
        pass


if __name__ == "__main__":
    reader = DataReader(data_path="../../data/", out_path="../../out/")
    data = reader.read(sub_path="melanoma/", batch_size = 10, is_shuffle = True, is_masked = False)

    plt.imshow(data[0])
    plt.show()
