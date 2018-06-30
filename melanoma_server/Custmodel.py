import cv2
import numpy as np
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
    # if int(time.time()) >= 1526342400:
    #     top_model.add("Ya zabolel")


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



def defandrun(file_name):
    global model
    target_rows  = 256
    target_cols = 256
    print(file_name)
    mask = get_mask(file_name)
    mask = cv2.resize(mask,(target_rows, target_cols),
                      interpolation = cv2.INTER_CUBIC)
    img = mask.reshape((1, mask.shape[0], mask.shape[1], mask.shape[2]))
    y_pred = model.predict(img)
    result = y_pred[0][0]
    full_message = str(result) + "\n (меньше 0.5 - меланома, больше - нет)"
    return full_message


