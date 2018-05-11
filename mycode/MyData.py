from mycode.FolderInspector import FolderInspector
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img, array_to_img
from keras.regularizers import l2
from keras.applications.densenet import preprocess_input
from keras.utils import plot_model,to_categorical
import numpy as np
from sklearn.utils import shuffle
from keras.applications.densenet import DenseNet121
from tqdm import tqdm
import gc



class MyData:
    def __init__(self, datapath = "./data"):
        self.path = datapath
        self.insp = FolderInspector()
        self.ROWS =256
        self.COLS = 256
        self.batch = -1 #if -1, this mean all data
        self.out_path = "./data/out/"

    def _image_array(self, file_path, files_list,  count):
        data = np.ndarray((count, self.ROWS, self.COLS, 3))
        print("Start reading your data from {} and make array ...".format(file_path))
        for i, image_file in tqdm(enumerate(files_list)):
            raw_img = load_img(file_path + image_file, target_size = (self.ROWS, self.COLS))
            img = img_to_array(raw_img)
            data[i] = img
        return data


    def _image_array_one(self, file_path):
        raw_img = load_img(file_path, target_size=(self.ROWS, self.COLS))
        img = img_to_array(raw_img)
        return img


    def _save(self, path, name, nparray):
        np.save(path + name + '.npy', nparray)


    def _split_ttstval(self, nparray):
        X_train, X_test, _, _ = train_test_split(nparray, nparray, test_size=0.2, random_state=1)
        X_train, X_val, _, _ = train_test_split(X_train, X_train, test_size=0.2, random_state=1)
        return X_train, X_test, X_val


    def _prepare_train_validate(self):
        #todo split to test/train and reshape to 256 to 256
        benign_fldr = self.path + "benign_mask/"
        melanoma_fldr = self.path + "melanoma_mask/"

        benign_files = self.insp.get_files(benign_fldr)
        melanoma_files = self.insp.get_files(melanoma_fldr)


        benign_array = self._image_array(benign_fldr, benign_files[:self.batch], len(benign_files[:self.batch]))
        melanoma_array = self._image_array(melanoma_fldr, melanoma_files[:self.batch],  len(melanoma_files[:self.batch]))
        print('End reading your data ...')

        benign_train, benign_test, benign_eval = self._split_ttstval(benign_array)
        melanoma_train, melanoma_test, melanoma_eval = self._split_ttstval(melanoma_array)

        del benign_array, melanoma_array
        gc.collect()

        return [benign_train, benign_test, benign_eval,
                melanoma_train, melanoma_test, melanoma_eval]

    def _train_base(self):

        print('Start preparing your data ...')
        list_array = self._prepare_train_validate()


        # for i, element in tqdm(enumerate(list_array)):
        #     self._save(self.out_path + "temp_path/", "element{}.pkl".format(i), element)


        #del list_array
        gc.collect()
        #list_array = list()
        # for
        # with open(self.out_path + "all_list.pkl", 'rb') as f:
        #     list_array = pickle.load(f)

        X_train = np.concatenate((list_array[0], list_array[3]), axis=0)
        len_first_part = len(list_array[0])
        len_sec_part = len(list_array[3])
        y_train = np.concatenate((np.ones((len_first_part,)),
                                  np.zeros((len_sec_part, ))),
                                  axis = 0)

        X_test = np.concatenate((list_array[1], list_array[4]), axis=0)
        len_first_part = len(list_array[1])
        len_sec_part = len(list_array[4])
        y_test = np.concatenate((np.ones((len_first_part,)),
                                 np.zeros((len_sec_part,))),
                                axis=0)

        X_val = np.concatenate((list_array[2], list_array[5]), axis=0)
        len_first_part = len(list_array[2])
        len_sec_part = len(list_array[5])
        y_val = np.concatenate((np.ones((len_first_part,)),
                                np.zeros((len_sec_part,))),
                               axis=0)


        train_preprocessed = preprocess_input(X_train)
        validation_preprocessed = preprocess_input(X_val)
        test_preprocessed = preprocess_input(X_test)

        print('Start training net ...')


        base_model = DenseNet121(input_shape=(256, 256, 3),
                                 include_top=False,
                                 weights='imagenet')

        ### Train Bottlenecks

        train_data = base_model.predict(train_preprocessed, verbose=1)

        validation_data = base_model.predict(validation_preprocessed, verbose=1)

        test_data = base_model.predict(test_preprocessed, verbose=1)

        # ### one-hot encoding of y-labels, first column is benign, second is melanoma, 3rd is sk
        # train_target = to_categorical(y_train, num_classes=3)
        # validation_target = to_categorical(y_val, num_classes=3)

        X_train_features, y_train = shuffle(train_data, y_train, random_state=10)
        X_val_features, y_val_features = shuffle(validation_data, y_val, random_state=10)


        self._save(self.out_path, "X_train", X_train_features)
        self._save(self.out_path, "y_train", y_train)

        self._save(self.out_path, "X_val_features", X_val_features)
        self._save(self.out_path, "X_val", X_val)
        self._save(self.out_path, "y_val_features", y_val_features)
        self._save(self.out_path, "y_val", y_val)

        del X_train_features, y_train, X_val_features, X_val, y_val_features, y_val


        #model_weights = base_model.get_weights()
        #model_architecture = base_model.to_json()
        #model_dict = {'model_weights': model_weights, 'model_architecture': model_architecture}

        # serialize model to JSON
        model_json = base_model.to_json()
        with open(self.out_path + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        base_model.save_weights(self.out_path + "model.h5")
        print("Saved model to disk")

        #with open(self.out_path + 'base_model.pickle', 'wb') as handle:
        #    pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del base_model
        gc.collect()
