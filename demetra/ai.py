from tensorflow_examples.models.pix2pix import pix2pix

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


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    dice_total = 0
    for idx in range(1, config.CLASSES):
        partial_truth = tf.cast(tf.equal(y_true, idx), tf.float32)
        dice_total += dice_coef(partial_truth, y_pred[..., idx])

    return 1 - (dice_total/(config.CLASSES - 1))


def get_model_p2pUnet(output_channels:int):
    """ """
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = True

    up_stack = [
        pix2pix.upsample(512, 3, apply_dropout=True),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3, apply_dropout=True),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3, apply_dropout=True),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3, apply_dropout=True),   # 32x32 -> 64x64
    ]

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


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


CALLBACKS = [tf.keras.callbacks.ModelCheckpoint(filepath='demetra_checkpoint', monitor='val_iou_score', mode='max', save_best_only=True)]


def add_sample_weights(image, label):
  class_weights = tf.constant([1.0, 1.0, 2.0])
  class_weights = class_weights/tf.reduce_sum(class_weights)
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  return image, label, sample_weights


def train_new_model(model_path, output_classes, epochs, batch_size=64):
    
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
    
    # show_predictions(model, test_batches, 5)

    model.save(model_path)


def train_current_model(model_path, epochs, batch_size=64, lr=0.0001):
    
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
    
    # show_predictions(model, test_batches, 5)

    model.save(model_path)

