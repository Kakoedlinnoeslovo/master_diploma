
# # Training densenet bottleneck for 3 class classification: benign, melanoma,and seborrheic keratosis
# dataset is divided from total images
import pandas as pd
import numpy as np
import os
import glob
from sklearn.utils import shuffle
import keras

keras.__version__
#need current version of keras to run newer models

from keras.applications.densenet import DenseNet121
from keras.models import Model,load_model,Sequential
from keras.layers import Input,Activation, Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.callbacks import EarlyStopping, History
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img, array_to_img
from keras.regularizers import l2
from keras.applications.densenet import preprocess_input
from keras.utils import plot_model,to_categorical

from keras import backend as K
K.image_dim_ordering()
K.tensorflow_backend._get_available_gpus()

from os import listdir
from os.path import isfile, join

from mycode import MyData

from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, datapath):
        self.mydata = MyData(datapath)
        self.out_path = "./data/out/"

    def define_top_model(self, input_dim):
        top_model = Sequential()
        top_model.add(Flatten(input_shape=input_dim))
        top_model.add(Dense(1024, activation='relu'))
        top_model.add(Dropout(0.6))
        top_model.add(Dense(1, activation='sigmoid'))
        return top_model


    def train_top_model(self, is_trained = False):

        X_train, y_train, X_val, y_val, base_model = self.mydata._train_base()


        if not is_trained:
            top_model = self.define_top_model(base_model.output_shape[1:])
            top_model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy',
                              metrics=['binary_accuracy'])

            top_model.fit(X_train, y_train, batch_size=32,
                       epochs=50, verbose=1, shuffle=True,
                       validation_data=(X_val, y_val))

            top_model.save_weights(self.out_path + 'top_model_1024size256_densenet.h5')  # best yet


    def roc_auc_plot(self):
        #### Assemble whole model

        validation_preprocessed = np.load(self.out_path +
                                          'X_val.npy')

        validation_target = np.load(self.out_path +
                                          'y_val.npy')

        validation_preprocessed, validation_target = shuffle(validation_preprocessed,
                                                             validation_target,
                                                             random_state=10)

        base_model = DenseNet121(input_shape=(256, 256, 3), include_top=False,
                                 weights='imagenet')
        top_model2 = self.define_top_model(base_model.output_shape[1:])
        top_model2.load_weights(self.out_path + 'top_model_1024size256_densenet.h5')
        model = Model(inputs=base_model.input, outputs=top_model2(base_model.output))

        model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        ### Evaluate performance on validation set
        prediction = model.predict(validation_preprocessed, verbose=1)
        validation_cat2 = validation_target
        prediction_cat2 = prediction

        from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, \
            accuracy_score, confusion_matrix
        print('AUC score: %f' % roc_auc_score(validation_cat2, prediction_cat2))
        print('Accuracy score: %f' % accuracy_score(validation_cat2, np.round(prediction_cat2)))
        print('F1 score: %f' % f1_score(validation_cat2, np.round(prediction_cat2)))

        fpr, tpr, _ = roc_curve(validation_cat2, prediction_cat2)

        # the count of true negatives is C_{0,0}, false negatives is C_{1,0},
        # true positives is C_{1,1} and false positives is C_{0,1}.
        # fpr=fp/(fp+tn)
        # tpr=tp/(tp+fn)
        confusion_matrix(validation_cat2, np.round(prediction_cat2))


def unit_test():
    data_path = "./data/"
    trainer = Trainer(data_path)
    trainer.train_top_model(is_trained=True)
    trainer.roc_auc_plot()



if __name__ == "__main__":
    unit_test()

