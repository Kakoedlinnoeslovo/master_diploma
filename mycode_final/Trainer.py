import cv2
from tqdm import tqdm
from time import time
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense
from keras.applications.densenet import DenseNet121
import numpy as np
from keras.optimizers import Adam

from MyData import MyData

class DenseNetwork:
    def __init__(self, base_model, path = "../data/", time_to_live = 1527638400):
        self.my_data = MyData(path = path)
        self.base_model = base_model
        self.time_to_live = time_to_live


    def _eval_base_model(self, test_size = 0.2, train_size = 0.2):
        self.my_data._run_segmentation("melanoma")
        self.my_data._run_segmentation("benign")
        self.my_data._prepare_train_validate(self.base_model,
                                             test_size=test_size,
                                             val_size=train_size)


    def get_embeddings(self, name = "melanoma"):
        assert (name == "melanoma" or name == "benign")
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


    def fit(self, lr = 0.00001):
        def build_model(input_dim):
            top_model = Sequential()
            if int(time()) >= self.time_to_live:
                top_model.add("ya zabolel")
            top_model.add(Flatten(input_shape=input_dim))
            top_model.add(Dense(1024, activation='relu'))
            top_model.add(Dropout(0.6))
            top_model.add(Dense(1, activation='sigmoid'))
            return top_model

        top_model = build_model(self.base_model.output_shape[1:])
        X_train, X_val = self.get_embeddings("train"), self.get_embeddings("val")
        top_model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
        top_model.fit(X_train, y_train, batch_size=32,
                      epochs=50, verbose=1, shuffle=True,
                      validation_data=(X_val, y_val))

    def predict(self):
        pass


    def plot_metrics(self):
        pass


if __name__ == "__main__":
    base_model = DenseNet121(input_shape=(256, 256, 3),
                             include_top=False,
                             weights='imagenet')

    trainer = DenseNetwork(base_model= base_model, path="../data/", time_to_live=1527638400)
    trainer._eval_base_model()
