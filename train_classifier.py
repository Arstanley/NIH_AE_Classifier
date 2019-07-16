from models.keras import ModelFactory
import argparse 
from generator import AugmentedImageSequence
import os
from utility import get_sample_counts
from weights import get_class_weights
from keras.utils import multi_gpu_model
from ClassificationCallback.callback import MultipleClassAUROC, MultiGPUModelCheckpoint
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--AE', type=bool, default=False)
    args = parser.parse_args()

    # Set Parameter # 
    base_model_name = "DenseNet121"
    use_base_model_weights = True
    weights_path = None
    image_dimension = 224
    batch_size = 32 
    epochs = 20
    class_names = ["Nodule", "Pneumothorax"]
    csv_path = './data/classification'
    image_source_dir = '/media/nfs/CXR/NIH/chest_xrays/NIH/data/images_1024x1024/'
    augmenter = None
    #  If train_steps is set to None, will calculate train steps by len(train)/batch_size
    train_steps = None #####   
    positive_weights_multiply = 1 
    outputs_path = './experiments/original'
    weights_name = f'weights.h5'
    output_weights_path = os.path.join(outputs_path, weights_name)
    initial_learning_rate = 0.0001
    training_stats = {}

    # Get Sample and Total Count From Training Data and Compute Class Weights # 
    train_counts, train_pos_counts = get_sample_counts(csv_path, "train", class_names) 
    if train_steps == None: 
        train_steps = int(train_counts / batch_size) 
    dev_counts, _ = get_sample_counts(csv_path, "test", class_names)
    validation_steps = int(dev_counts / batch_size)
    print ('***Compute Class Weights***')
    class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply)
    print(class_weights)
    
    # Create Image Sequence # 
    
    train_sequence = AugmentedImageSequence(
            dataset_csv_file=os.path.join(csv_path, "train.csv"),
            class_names=class_names,
            source_image_dir=image_source_dir,
            batch_size=batch_size,
            target_size=(image_dimension, image_dimension),
            augmenter=augmenter,
            steps=train_steps)

    validation_sequence = AugmentedImageSequence(
            dataset_csv_file=os.path.join(csv_path, "test.csv"),
            class_names = class_names,
            source_image_dir = image_source_dir,
            batch_size = batch_size,
            target_size = (image_dimension, image_dimension),
            augmenter = augmenter,
            steps = validation_steps, 
            shuffle_on_epoch_end=False)
    
    # Build Model # 
    factory = ModelFactory()
    model = factory.get_model(class_names,
            model_name=base_model_name,
            use_base_weights=use_base_model_weights,
            weights_path=None,
            input_shape=(image_dimension, image_dimension, 3))

    print("** check multiple gpu availability **")
    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
    if gpus > 1:
        print("** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
        # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
        checkpoint = MultiGPUModelCheckpoint(
            filepath=output_weights_path,
            base_model=model,
        )
    else:
        model_train = model
        checkpoint = ModelCheckpoint(
             output_weights_path,
             save_weights_only=True,
             save_best_only=True,
             verbose=1,
        )


    auroc = MultipleClassAUROC(
            sequence=validation_sequence,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
            workers=8
        )
    callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(outputs_path, "logs"), batch_size=batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1,
                              verbose=1, mode="min", min_lr=1e-8),
            auroc,
        ]

    # Compile Model # 
    print ('*** Start Compiling ***')
    optimizer = Adam(lr=initial_learning_rate)
    model_train.compile(optimizer=optimizer, loss="binary_crossentropy")

    # Train # 
    print("** start training **")
    history = model_train.fit_generator(
        generator=train_sequence,
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=validation_sequence,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        workers=8,
        shuffle=False,
    )
    # dump history
    print("** dump history **")
    with open(os.path.join(outputs_path, "history.pkl"), "wb") as f:
        pickle.dump({
            "history": history.history,
            "auroc": auroc.aurocs,
        }, f)
    print("** done! **")

if __name__ == "__main__":
    main()
