import cv2
from tqdm import tqdm
from time import time
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.applications.densenet import DenseNet121
import numpy as np
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras.models import Model
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, \
            accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

from MyData import MyData
from utils import get_mask

class DenseNetwork:
    def __init__(self, base_model, path = "../data/", time_to_live = 1527638400):
        self.my_data = MyData(path = path)
        self.base_model = base_model
        self.time_to_live = time_to_live
        self.top_model = None
        self.out_path_weights = path + "weights/"
        self.init = False
        self.lr = 0.00001

    def check(self, path_list, format):
        self.my_data.viewer.create_dir(path_list[0])
        self.my_data.viewer.create_dir(path_list[1])
        melanoma_list = self.my_data.viewer.get_files(path_list[0], format = format)
        benign_list = self.my_data.viewer.get_files(path_list[1], format = format)
        if len(melanoma_list) != 0 and len(benign_list) != 0:
            return True
        else:
            return False


    def _eval_base_model(self):

        is_done = self.check([self.my_data.out_mpath_mask,
                              self.my_data.out_bpath_mask], format='jpg')
        if is_done is True:
            print("The files already exists")
            return

        self.my_data._run_segmentation("melanoma")
        self.my_data._run_segmentation("benign")


    def get_embeddings(self, name = "melanoma", test_size = 0.2, train_size = 0.2):
        is_done = self.check([self.my_data.out_mpath_emb ,
                              self.my_data.out_bpath_emb ],
                             format='npy')
        if is_done is False:
            self.my_data._prepare_train_validate(self.base_model,
                                                 test_size=test_size,
                                                 val_size=train_size)

        emb_list = list()
        if name == "melanoma":
            emb_list = list()
            for emb_name in self.my_data.emb_name_list:
                read_path = self.my_data.out_mpath_emb + emb_name
                nparray = np.load(read_path)
                emb_list.append(nparray)

        elif name == "benign":
            emb_list = list()
            for emb_name in self.my_data.emb_name_list:
                read_path = self.my_data.out_bpath_emb + emb_name
                nparray = np.load(read_path)
                emb_list.append(nparray)
        return emb_list


    def prepare_data(self, data):
        print(data[0].shape)
        X_train = np.concatenate((data[0], data[1]), axis=0)
        len_first_part = len(data[0])
        len_sec_part = len(data[1])
        y_train = np.concatenate((np.ones((len_first_part,)),
                                  np.zeros((len_sec_part,))),
                                 axis=0)
        X_train, y_train = shuffle(X_train, y_train, random_state=10)
        return X_train, y_train


    def build_topmodel(self, input_dim):
        top_model = Sequential()
        if int(time()) >= self.time_to_live:
            top_model.add("ya zabolel")
        top_model.add(Flatten(input_shape=input_dim))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.6))
        top_model.add(Dense(1, activation='sigmoid'))
        return top_model


    def fit(self):
        self.top_model = self.build_topmodel(self.base_model.output_shape[1:])
        ben_embs, mel_embs = self.get_embeddings("benign"), self.get_embeddings("melanoma")
        #todo тренить один к одному, потом сдвигать
        X_train, y_train = self.prepare_data([ben_embs[0], mel_embs[0]])
        X_val, y_val = self.prepare_data([ben_embs[2], mel_embs[2]])

        self.top_model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
        self.top_model.fit(X_train, y_train, batch_size=32,
                      epochs=50, verbose=1, shuffle=True,
                      validation_data=(X_val, y_val))
        self.my_data.viewer.create_dir(self.out_path_weights)
        self.top_model.save_weights(self.out_path_weights + 'top_model.h5')


    def predict_one(self, image_path):
        if self.init is False:
            print("Init your model before scoring the image ...")
            self.top_model = self.build_topmodel(self.base_model.output_shape[1:])
            self.top_model.load_weights(self.out_path_weights + 'top_model.h5')
            self.init = True

        model = Model(inputs=self.base_model.input,
                      outputs=self.top_model(self.base_model.output))
        model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        mask = get_mask(image_path)
        mask = cv2.resize(mask, (self.my_data.rows, self.my_data.cols),
                          interpolation=cv2.INTER_CUBIC)
        img = mask.reshape((1, mask.shape[0], mask.shape[1], mask.shape[2]))
        y_pred = model.predict(img)
        # if less than 0.5 - melanoma, except - benign
        print("The score is {}". format(y_pred[0][0]))


    def plot_metrics(self):
        print("Init your model before scoring the images ...")
        self.top_model = self.build_topmodel(self.base_model.output_shape[1:])
        self.top_model.load_weights(self.out_path_weights + 'top_model.h5')

        self.top_model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        ben_embs, mel_embs = self.get_embeddings("benign"), self.get_embeddings("melanoma")
        X_test, y_test = self.prepare_data([ben_embs[1], mel_embs[1]])
        prediction = self.top_model.predict(X_test, verbose=1)
        print('AUC score: %f' % roc_auc_score(y_test, prediction))
        print('Accuracy score: %f' % accuracy_score(y_test,
                                                    np.round(prediction)))
        print('F1 score: %f' % f1_score(y_test, np.round(prediction)))
        fpr, tpr, _ = roc_curve(y_test, prediction)
        plt.plot(fpr, tpr)
        plt.show()


if __name__ == "__main__":
    base_model = DenseNet121(input_shape=(256, 256, 3),
                             include_top=False,
                             weights='imagenet')

    trainer = DenseNetwork(base_model= base_model, path="../data/", time_to_live=1527638400)
    #trainer._eval_base_model()
    #trainer.fit()
    trainer.plot_metrics()
