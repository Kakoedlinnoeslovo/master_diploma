from .TrainerDetector import DenseNetwork
from .utils import double_conv_layer

from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras import optimizers
from keras.applications.densenet import DenseNet121
from keras import losses
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from keras.optimizers import Adam



class SegmentorDetector(DenseNetwork):
	def __init__(self, base_model, path="../data/", time_to_live=1527638400):
		super().__init__(base_model, path, time_to_live)
		self.out_path_numpy = path + "out/numpy/"


	def _build_snet(self, im_size=256, batch_norm=True):
		inputs = Input((im_size, im_size, 3))
		axis = 3
		filters = 32
		mask_channels = 3

		conv_224 = double_conv_layer(inputs, filters, 0, batch_norm)
		pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
		conv_7 = double_conv_layer(pool_112, 2 * filters, 0, batch_norm)

		up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_224], axis=axis)
		up_conv_14 = double_conv_layer(up_14, 2 * filters, 0, batch_norm)

		conv_final = Conv2D(mask_channels, (1, 1))(up_conv_14)
		conv_final = Activation('sigmoid', name = "deep_features")(conv_final)

		model = Model(inputs, conv_final, name="ZF_UNET_224")
		return model


	def build_full_model(self):
		snet = self._build_snet()

		# set layers of base model non_trainable
		for layer in self.base_model.layers:
			layer.trainable = False

		self.top_model = self.build_topmodel(self.base_model.output_shape[1:])

		model = Model(inputs=snet.input,
		              outputs=self.top_model(self.base_model(snet.output)),
		              name="SegmentorDetector")
		return model


	def read_data(self, folder_name="melanoma"):
		if folder_name == "melanoma":
			files = self.my_data.viewer.get_files(self.my_data.in_mpath, format='jpg')
			temp_inpath = self.my_data.in_mpath
		elif folder_name == "benign":
			files = self.my_data.viewer.get_files(self.my_data.in_bpath, format='jpg')
			temp_inpath = self.my_data.in_bpath
		else:
			print("folder_name should be melanoma or benign")
			return
		data = np.zeros((len(files), self.my_data.rows, self.my_data.cols, 3))
		for i, file in enumerate(tqdm(files)):
			# image = cv2.imread(temp_inpath + file)
			raw_img = load_img(temp_inpath + file, target_size=(self.my_data.rows, self.my_data.cols))
			img = img_to_array(raw_img)
			data[i] = img
		return data



	def show_result(self):
		img = cv2.imread("../data/melanoma/0000.jpg")
		model = self.build_full_model()
		img = cv2.resize(img, (self.my_data.rows, self.my_data.cols),
		                  interpolation=cv2.INTER_CUBIC)
		img = img.reshape(1, self.my_data.rows, self.my_data.cols, 3)
		model.load_weights(self.out_path_weights + 'model_seg.h5')
		reduced_model = Model(inputs=model.input, outputs=model.get_layer('deep_features').output)
		mask = reduced_model.predict(img)
		mask = mask.reshape(self.my_data.rows, self.my_data.cols, 3)
		plt.imshow(mask)
		plt.show()


	def fit(self, lr=0.0001):
		check = self.check(self.out_path_numpy, format='npy')
		if check is False:
			benign_data = self.read_data("benign")
			melanoma_data = self.read_data("melanoma")
			X, y = self.prepare_data([benign_data, melanoma_data])
			self.my_data.viewer.create_dir(self.out_path_numpy)
			np.save(self.out_path_numpy + "X.npy", X)
			np.save(self.out_path_numpy + "y.npy", y)
		else:
			X, y = np.load(self.out_path_numpy + "X.npy"), np.load(self.out_path_numpy + "y.npy")

		X_train, X_val, y_train, y_val = train_test_split(X, y,
		                                                  test_size=0.1,
		                                                  random_state=42)
		model = self.build_full_model()
		model.compile(optimizers.Adam(lr=lr),
		              loss=losses.binary_crossentropy,
		              metrics=['binary_accuracy'])
		model.summary()
		self.my_data.viewer.create_dir(self.out_path_weights)
		filepath = self.out_path_weights + "model_seg.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
		                             save_best_only=True, mode='max')
		callbacks_list = [checkpoint]
		model.fit(X_train, y_train, batch_size=10,
		          epochs=50, verbose=1, shuffle=True,
		          validation_data=(X_val, y_val), callbacks=callbacks_list)
		self.my_data.viewer.create_dir(self.out_path_weights)
		model.save_weights(self.out_path_weights + 'model_seg.h5')

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




def unit_test():
	base_model = DenseNet121(input_shape=(256, 256, 3),
	                         include_top=False,
	                         weights='imagenet')
	detector = SegmentorDetector(base_model)
	#detector.fit()
	detector.show_result()


if __name__ == "__main__":
	unit_test()
