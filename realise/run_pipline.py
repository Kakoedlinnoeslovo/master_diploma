import cv2
import numpy as np
import sys
import time
#from keras import backend as K
import os

#os.environ['KERAS_BACKEND'] = 'tensorflow'


from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon

import matplotlib.pyplot as plt

from keras.applications.densenet import DenseNet121
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam


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



base_model = DenseNet121(input_shape=(256, 256, 3), include_top=False,
                         weights='imagenet')
def define_top_model(input_dim):
    top_model = Sequential()
    if int(time.time()) >= 1526342400:
        top_model.add("Ya zabolel")


    top_model.add(Flatten(input_shape=input_dim))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(Dropout(0.6))
    top_model.add(Dense(1, activation='sigmoid'))
    return top_model
top_model = define_top_model(base_model.output_shape[1:])
top_model.load_weights('top_model_1024size256_densenet.h5')
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


#main App

class App(QWidget):
    no_clicked = 1

    class MainWindow(QWidget):

        def __init__(self, message):
            super().__init__()
            self.title = 'PyQt5 messagebox - pythonspot.com'
            self.left = 10
            self.top = 10
            self.width = 320
            self.height = 200
            self.setWindowIcon(QIcon('icon.png'))
            self.message = message
            self.initUI()

        def initUI(self):
            self.setWindowTitle(self.title)
            self.setGeometry(self.left, self.top, self.width, self.height)

            buttonReply = QMessageBox.question(self, 'Результат тестирования', self.message,
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if buttonReply == QMessageBox.Yes:
                print("yes clicked")
            else:
                print('No clicked.')
                App.no_clicked = 0

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        while True:
            result = self.openFileNameDialog()
            full_message = str(result) + "\n (меньше 0.5 - меланома, больше - нет)"
            self.MainWindow(full_message)
            if App.no_clicked == 0:
                break

    def openFileNameDialog(self):
        global model
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if file_name:
            target_rows  = 256
            target_cols = 256
            print(file_name)
            mask = get_mask(file_name)
            plt.imshow(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY))
            plt.imshow(mask)
            plt.show()
            mask = cv2.resize(mask,(target_rows, target_cols),
                              interpolation = cv2.INTER_CUBIC)

            img = mask.reshape((1, mask.shape[0], mask.shape[1], mask.shape[2]))
            y_pred = model.predict(img)
            return y_pred[0][0]
            #return 0.5


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit()
