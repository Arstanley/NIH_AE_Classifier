from CNN_AE import CNN_AE 
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd 
import os
from callbacks import SaveImageNModel
import keras.backend as K
from argparse import ArgumentParser
import numpy as np
import cv2

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true)))

def select_normal(df):
    return df[df["No Finding"] == 1]

def main():
    ### Parse Arguments ### 
    parser = ArgumentParser()
    parser.add_argument('--csv_path', type=str)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--padding', type=str, default='same') 
    parser.add_argument('--imagedir', type=str, default='/media/nfs/CXR/NIH/chest_xrays/NIH/data/images_1024x1024/')
    args = parser.parse_args()
    print(args)

    ### Load the data frame and Image Generator ### 
    train_path = os.path.join(args.csv_path, 'train.csv')
    valid_path = os.path.join(args.csv_path, 'test.csv')
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    ### Select Normal Images ### 
    train = select_normal(train)
    valid = select_normal(valid)

    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_dataframe(
            dataframe=train, 
            directory=args.imagedir,
            x_col='Image Index', 
            class_mode='input', 
            batch_size=32, 
            color_mode="grayscale", 
            target_size = (224, 224))
    valid_generator = datagen.flow_from_dataframe(
            dataframe=valid, 
            directory=args.imagedir,
            x_col='Image Index', 
            class_mode='input', 
            batch_size=32, 
            color_mode="grayscale", 
            target_size = (224, 224))
    
    ### Load Some Sample Image ### 
    from glob import glob
    from skimage.io import imread
    imgs = []
    imgs_sample = train.sample(10)
    for idx, row in imgs_sample.iterrows():
        image = cv2.imread("/media/nfs/CXR/NIH/chest_xrays/NIH/data/images_1024x1024/"+imgs_sample.loc[idx, "Image Index"], 0)
        image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_CUBIC)
        image = image / 255.
        imgs.append(np.array(image))
    imgs = np.array(imgs)
    print(imgs.shape)
    
    ### Train model ### 
    AE = CNN_AE()
    model = AE.get_model()
    model.compile(loss = root_mean_squared_error, optimizer = 'adadelta', metrics=['accuracy'])
    print(model.summary)
    modelCallback = SaveImageNModel(imgs)
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=250, 
                        epochs = args.epoch, 
                        shuffle=True, 
                        callbacks = [modelCallback], 
                        validation_data=valid_generator, 
                        validation_steps = 15)

if __name__ == '__main__':
    main()
