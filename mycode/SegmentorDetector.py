from TrainerDetector import DenseNetwork
from utils import double_conv_layer, dice_coef_loss

from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras import optimizers
from keras.applications.densenet import DenseNet121
from keras import losses

class SegmentorDetector(DenseNetwork):
    def __init__(self, base_model, path = "../data", time_to_live = 1527638400):
        super().__init__(base_model, path, time_to_live)


    def _build_snet(self, im_size = 256, batch_norm=True):
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
        conv_final = Activation('sigmoid')(conv_final)


        model = Model(inputs, conv_final, name="ZF_UNET_224")
        return model


    def build_full_model(self):
        snet = self._build_snet()

        #set layers of base model non_trainable
        for layer in self.base_model.layers:
            layer.trainable = False

        self.top_model = self.build_topmodel(self.base_model.output_shape[1:])

        model = Model(inputs=snet.input,
                      outputs=self.top_model(self.base_model(snet.output)),
                      name = "SegmentorDetector")
        model.compile(optimizers.Adam(lr = 0.0001),
                      loss=losses.binary_crossentropy)
        model.summary()


def unit_test():
    base_model = DenseNet121(input_shape=(256, 256, 3),
                             include_top=False,
                             weights='imagenet')
    detector = SegmentorDetector(base_model)
    detector.build_full_model()

if __name__ == "__main__":
    unit_test()
