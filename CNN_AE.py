import keras
from keras.layers import Dense, Input, Lambda, Reshape, Dropout, Flatten, Activation
import numpy as np
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D

class CNN_AE:
    def __init__(self, dp_rate=0.4, input_shape = (224, 224, 1), padding = 'same'):
        self.input_shape = input_shape
        self.padding = padding
        self.dp_rate = dp_rate
    def get_model(self):
        model = keras.Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3), activation = 'relu', input_shape=self.input_shape, padding = self.padding))
        keras.layers.core.Dropout(self.dp_rate, noise_shape=None, seed=None)
        model.add(MaxPooling2D(pool_size=(2, 2), padding = self.padding))
        model.add(Conv2D(8, kernel_size = (3, 3), activation = 'relu', padding = self.padding))
        keras.layers.core.Dropout(self.dp_rate, noise_shape=None, seed=None)
        model.add(MaxPooling2D(pool_size=(2, 2), padding = self.padding))
        model.add(Conv2D(4, kernel_size = (3, 3), activation = 'relu', padding = self.padding))
        keras.layers.core.Dropout(self.dp_rate, noise_shape=None, seed=None)
        model.add(MaxPooling2D(pool_size = (2, 2), padding = self.padding))

        # Decoder
        model.add(Conv2D(4, kernel_size = (3, 3), activation = 'relu', padding = self.padding))
        keras.layers.core.Dropout(self.dp_rate, noise_shape=None, seed=None)
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(8, kernel_size = (3, 3), activation = 'relu', padding = self.padding))
        keras.layers.core.Dropout(self.dp_rate, noise_shape=None, seed=None)
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu', padding = self.padding))
        keras.layers.core.Dropout(self.dp_rate, noise_shape=None, seed=None)
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, kernel_size = (3, 3), activation = 'sigmoid', padding = self.padding))
        return model 
