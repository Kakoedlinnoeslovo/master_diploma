from .utils import get_mask
from .utils import FolderViewer

from tqdm import tqdm
import cv2
import gc
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.applications.densenet import preprocess_input


class MyData:
    def __init__(self, path = "../data/"):
        self.viewer = FolderViewer()
        out = path + "out/"
        self.in_mpath = path + "melanoma/"
        self.in_bpath = path + "benign/"

        self.out_mpath_mask = out + 'melanoma_mask/'
        self.out_bpath_mask  = out + 'benign_mask/'

        self.out_mpath_emb = out + 'melanoma_embeddings/'
        self.out_bpath_emb = out + "benign_embeddings/"

        self.emb_name_list = ['train_emb.npy', 'test_emb.npy', 'val_emb.npy']
        self.cols, self.rows, self.channels = 256, 256, 3


    def _run_segmentation(self, folder_name = "melanoma"):
        if folder_name == "melanoma":
            files = self.viewer.get_files(self.in_mpath, format='jpg')
            temp_outpath = self.out_mpath_mask
            temp_inpath = self.in_mpath
        elif folder_name == "benign":
            files = self.viewer.get_files(self.in_bpath, format='jpg')
            temp_outpath = self.out_bpath_mask
            temp_inpath = self.in_bpath
        else:
            print("folder_name should be melanoma or benign")
            return
        self.viewer.create_dir(temp_outpath)
        for i, file in enumerate(tqdm(files)):
            mask = get_mask(temp_inpath + file)
            temp_path = temp_outpath + file
            if i % 100 == 0:
                print('\n' + temp_path)
            cv2.imwrite(temp_path, mask)


    def _prepare_train_validate(self, base_model, test_size = 0.2, val_size = 0.2):
        benign_fldr = self.out_bpath_mask
        melanoma_fldr = self.out_mpath_mask
        benign_files = self.viewer.get_files(benign_fldr, format='jpg')
        melanoma_files = self.viewer.get_files(melanoma_fldr, format='jpg')

        def image_array(file_path, files_list, quanity, rows, cols, channels):
            data = np.ndarray((quanity, rows, cols, channels))
            print("\nStart reading your data from {} and making array ...".format(file_path))
            for i, image_file in tqdm(enumerate(files_list)):
                raw_img = load_img(file_path + image_file, target_size=(rows, cols))
                img = img_to_array(raw_img)
                data[i] = img
            return data

        def split_train_test(nparray, test_size, val_size):
            X_train, X_test, _, _ = train_test_split(nparray, nparray,
                                                     test_size=test_size, random_state=1)
            X_train, X_val, _, _ = train_test_split(X_train, X_train,
                                                    test_size=val_size, random_state=1)
            return X_train, X_test, X_val

        def prepare_embeddings(base_model, viewer, train, test, val, out_emb, name_list):
            def save(path, name, nparray):
                np.save(path + name, nparray)

            train_prep, test_prep, val_prep  = preprocess_input(train), \
                                               preprocess_input(test), \
                                               preprocess_input(val)

            train_emb, test_emb, val_emb = base_model.predict(train_prep, verbose=1), \
                                           base_model.predict(test_prep, verbose=1),\
                                           base_model.predict(val_prep, verbose=1),

            #save files
            nparray_list = [train_emb, test_emb, val_emb]
            viewer.create_dir(out_emb)
            # assert(len(nparray_list) == len(name_list),
            #        'len(nparray_list) should equal to len(name_list)' )
            for nparray, name in zip(nparray_list, name_list):
                save(out_emb, name, nparray)


        benign_array = image_array(benign_fldr, benign_files,
                                   len(benign_files),
                                   self.rows, self.cols, self.channels)

        melanoma_array = image_array(melanoma_fldr, melanoma_files,
                                     len(melanoma_files),
                                     self.rows, self.cols, self.channels)
        print('\nEnd reading your data ...')
        benign_train, benign_test, benign_val = split_train_test(benign_array,
                                                                 test_size,
                                                                 val_size)
        melanoma_train, melanoma_test, melanoma_val = split_train_test(melanoma_array,
                                                                       test_size,
                                                                       val_size)
        del benign_array, melanoma_array
        gc.collect()

        print('\nStart preparing your embeddings ...')
        prepare_embeddings(base_model, self.viewer,
                           benign_train, benign_test,
                           benign_val, self.out_bpath_emb,
                           self.emb_name_list)
        prepare_embeddings(base_model, self.viewer,
                           melanoma_train,
                           melanoma_test,
                           melanoma_val, self.out_mpath_emb,
                           self.emb_name_list)
        print('\nDone')