import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import random
import config
import cv2
import os

os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
sm.set_framework('tf.keras')

from tensorflow import keras
keras.backend.set_image_data_format('channels_last')

K = tf.keras.backend
CALLBACKS = [tf.keras.callbacks.ModelCheckpoint(filepath='demetra_checkpoint', monitor='val_iou_score', mode='max', save_best_only=True)]

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=34):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.

        self.augment_inputs = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(1., fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.35, 0.35, fill_mode="reflect", seed=seed),
            tf.keras.layers.GaussianNoise(0.25, seed=seed),
            tf.keras.layers.RandomZoom(0.2, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.25, seed=seed)
        ])

        self.augment_labels = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(1., fill_mode="reflect", interpolation='nearest', seed=seed),
            tf.keras.layers.RandomTranslation(0.35, 0.35, fill_mode="reflect", interpolation='nearest', seed=seed),
            tf.keras.layers.RandomZoom(0.2, fill_mode="reflect", interpolation='nearest', seed=seed)
        ])


    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        labels = tf.cast(labels, tf.uint8)

        return inputs, labels


def get_dataset(images_path, masks_path):
    images = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    masks = [os.path.join(masks_path, f) for f in os.listdir(masks_path)]

    datapoints_total = len(images)

    train_images = tf.data.Dataset.from_tensor_slices((images, masks)).map(__get_datapoint_images, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = tf.data.Dataset.from_tensor_slices((images, masks)).map(__get_datapoint_images, num_parallel_calls=tf.data.AUTOTUNE)

    return datapoints_total, train_images, test_images


def __get_datapoint_images(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_bmp(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=(128, 128))

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_bmp(mask)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(images=mask, size=(128, 128), method='nearest')
    
    image, mask = __normalize(image, mask)

    return image, mask


def __normalize(image, mask):
    image = tf.cast(image, tf.float32) / 255.0
    return image, mask


def __to_categorical(image, mask):
    mask = tf.squeeze(mask, axis=-1) 
    mask = tf.one_hot(tf.cast(mask, tf.int32), 3)
    mask = tf.cast(mask, tf.float32)
    return image, mask


def display(display_list, tensors=True):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        if tensors:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        else:
            plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])


def add_sample_weights(image, label):
  class_weights = tf.constant([1.0, 1.0, 2.0])
  class_weights = class_weights/tf.reduce_sum(class_weights)
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  return image, label, sample_weights


def train_new_model(model_path, output_classes, epochs, batch_size=64):
    """
    Train a new U-Net model with EfficientNetB3 (by default) as the encoder.

    This function sets up training and validation datasets using images and masks from specified paths,
    applies data augmentation to the training dataset, compiles the model, and then trains it.
    The trained model is saved to the specified path.

    Parameters:
    - model_path (str): The path where the trained model should be saved.
    - output_classes (int): The number of classes for the output layer of the U-Net model.
    - epochs (int): The number of epochs to train the model.
    - batch_size (int, optional): The size of the batches of data (default is 64).

    The function retrieves the dataset paths from the configuration, prepares the datasets,
    defines the model architecture, compiles it, trains it, and finally saves the trained model.

    Note:
    - The training and validation datasets are extracted from `config.DATASET_FOLDER` combining `config.IMAGES_FOLDER`
      for images and `config.MASKS_FOLDER` for masks.
    - Data augmentation is applied to the training dataset.
    - The model uses the 'nadam' optimizer, a combination of Jaccard loss and Categorical Focal loss,
      and monitors the IoU score and F-score as metrics.

    The model is trained using the specified `epochs` and `batch_size`, and the progress is printed during training.
    After training, the model is saved to `model_path`.
    """
    
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

    datapoints_total, train_images, test_images = get_dataset(images_path, masks_path)

    train_batches = (
        train_images
        .cache()
        .shuffle(batch_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=random.randint(1,999))).map(__to_categorical)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size).map(__to_categorical)
    
    print("Train Dataset:", train_batches)
    print("Val Dataset:", test_batches)

    model = sm.Unet('efficientnetb3', classes=output_classes, encoder_weights='imagenet', activation='softmax')
    model.compile(optimizer='nadam',           
                loss=sm.losses.JaccardLoss() + sm.losses.CategoricalFocalLoss(),
                metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()])

    _ = model.fit(train_batches, epochs=epochs,
                          steps_per_epoch=int(datapoints_total/batch_size),
                          validation_steps=int(datapoints_total/batch_size),
                          validation_data=test_batches,
                          callbacks=[])

    model.save(model_path)


def train_current_model(model_path, epochs, batch_size=64, lr=0.0001):
    """
    Train an existing U-Net model by specifing the path to model files

    This function sets up training and validation datasets using images and masks from specified paths,
    applies data augmentation to the training dataset, compiles the model, and then trains it.
    The trained model is saved to the specified path.

    Parameters:
    - model_path (str): The path where the trained model should be saved.
    - output_classes (int): The number of classes for the output layer of the U-Net model.
    - epochs (int): The number of epochs to train the model.
    - lr (float, optional): The learning rate step for Adam or other optimizer.

    The function retrieves the dataset paths from the configuration, prepares the datasets,
    defines the model architecture, compiles it, trains it, and finally saves the trained model.

    Note:
    - The training and validation datasets are extracted from `config.DATASET_FOLDER` combining `config.IMAGES_FOLDER`
      for images and `config.MASKS_FOLDER` for masks.
    - Data augmentation is applied to the training dataset.
    - The model uses the 'adam' optimizer, a combination of Jaccard loss and Categorical Focal loss,
      and monitors the IoU score and F-score as metrics.

    The model is trained using the specified `epochs` and `batch_size`, and the progress is printed during training.
    After training, the model is saved to `model_path`.
    """
    
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

    datapoints_total, train_images, test_images = get_dataset(images_path, masks_path)

    train_batches = (
        train_images
        .cache()
        .shuffle(batch_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=random.randint(1, 999))).map(__to_categorical)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size).map(__to_categorical)
    
    print("Train Dataset:", train_batches)
    print("Val Dataset:", test_batches)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # 'adam',
                loss=sm.losses.JaccardLoss() + sm.losses.CategoricalFocalLoss(),
                metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()])

    _ = model.fit(train_batches, epochs=epochs,
                          steps_per_epoch=int(datapoints_total/batch_size),
                          validation_steps=int(datapoints_total/batch_size),
                          validation_data=test_batches,
                          callbacks=CALLBACKS)

    model.save(model_path)
