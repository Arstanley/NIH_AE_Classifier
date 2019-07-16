import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from PIL import Image
from skimage.transform import resize
from keras.models import model_from_json
import matplotlib.image as mpimg
import cv2
import scipy.misc

class AugmentedImageSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, dataset_csv_file, class_names, source_image_dir, batch_size=16,
                 target_size=(224, 224), augmenter=None, verbose=0, steps=None,
                 shuffle_on_epoch_end=True, random_state=1, model_epoch=12):

        ### Model_epoch parameter exists for choosing an autoencoder model based on the         ### give number of epoch
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param class_names: list of str
        :param batch_size: int
        :param target_size: tuple(int, int)
        :param augmenter: imgaug object. Do not specify resize in augmenter.
                          It will be done automatically according to input_shape of the model.
        :param verbose: int
        """
        self.dataset_df = pd.read_csv(dataset_csv_file)
        self.source_image_dir = source_image_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmenter = augmenter
        self.verbose = verbose
        self.shuffle = shuffle_on_epoch_end
        self.random_state = random_state
        self.class_names = class_names
        self.prepare_dataset()
        ### Get the model based on the epoch number ###
        json_file = open(f'./checkpoints/model_{model_epoch}.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        ### Load weights into the model ### 
        self.model.load_weights(f"./checkpoints/model_{model_epoch}.h5")
        self.model._make_predict_function()
        print(f"Autoencoder model (epoch = {model_epoch}) is loaded successfully")
        if steps is None:
            self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))
        else:
            self.steps = int(steps)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_image(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch_images(batch_x)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def load_image(self, image_file):
        image_path = os.path.join(self.source_image_dir, image_file)
        image_array = cv2.imread(image_path, 0)
        image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_CUBIC)
        image_array = np.array(image_array / 255.)
        prediction = self.model.predict(image_array.reshape(1, 224, 224, 1))
        diff_recons = image_array.reshape(224, 224) - prediction.reshape(224, 224)
        diff_recons = np.stack((diff_recons,)*3, axis=-1)
        diff_recons = resize(diff_recons, (224, 224))
        return diff_recons

    def transform_batch_images(self, batch_x):
        if self.augmenter is not None:
            batch_x = self.augmenter.augment_images(batch_x)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch_x = (batch_x - imagenet_mean) / imagenet_std
        return batch_x

    def get_y_true(self):
        """
        Use this function to get y_true for predict_generator
        In order to get correct y, you have to set shuffle_on_epoch_end=False.

        """
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        return self.y[:self.steps*self.batch_size, :]

    def prepare_dataset(self):
        df = self.dataset_df.sample(frac=1., random_state=self.random_state)
        df.fillna(0, inplace=True)
        self.x_path, self.y = df["Image Index"], df[self.class_names].as_matrix()

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.prepare_dataset()
