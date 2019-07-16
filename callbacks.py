from keras.callbacks import Callback
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class SaveImageNModel(Callback):
    def __init__(self, imgs, outputdir='./checkpoints', imagedir='./ae_samples'):
        # Pass sample images to the call back function 
        self.imgs = imgs
        self.outputdir = outputdir 
        self.imagedir = imagedir
    def on_epoch_end(self, epoch, logs={}):
        # Save Model
        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)
        with open(f"./checkpoints/model_{epoch}.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(f"./checkpoints/model_{epoch}.h5")
        print(f"Successfully saved Autoencoder #{epoch} to dir {self.outputdir}")
        # Save Images
        predictions = self.model.predict(self.imgs.reshape(self.imgs.shape[0], 224, 224, 1))
        plt.close('all') 
        fig, axs = plt.subplots(3, 10, figsize = (80, 40))
        mse = []
        for i in range(10):
            mse.append(mean_squared_error(self.imgs[i].reshape(224, 224), 
                predictions[i].reshape(224, 224)))
            axs[0, i].imshow(self.imgs[i].reshape(224, 224), cmap = "Greys_r")
            axs[1, i].imshow(predictions[i].reshape(224, 224), cmap = "Greys_r")
            axs[2, i].imshow(self.imgs[i].reshape(224, 224) - predictions[i].reshape(224, 224), cmap = "Greys_r")
        if not os.path.isdir(self.imagedir):
            os.makedirs(self.imagedir)
        fig.savefig(f'./{self.imagedir}/{epoch}.jpg', bbox_inches="tight")
        print(f'test_set_w/abnormal_mse: {np.mean(mse)}')

