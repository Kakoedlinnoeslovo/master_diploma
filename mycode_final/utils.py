import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
from tqdm import tqdm



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

def unit_test():
    path = "../data/benign/0000.jpg"
    img = get_mask(path)
    plt.imshow(img)
    plt.show()
    cv2.imwrite("../data/test/0000.jpg", img)

if __name__ == "__main__":
    #unit_test()
    _run_segmentation('melanoma')