from tensorflow_examples.models.pix2pix import pix2pix

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import random
import config
import cv2
import os

K = tf.keras.backend


class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=76):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.

        self.augment_inputs = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(1., fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.25, 0.25, fill_mode="reflect", seed=seed)
        ])

        self.augment_labels = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(1., fill_mode="reflect", interpolation='nearest', seed=seed),
            tf.keras.layers.RandomTranslation(0.25, 0.25, fill_mode="reflect", interpolation='nearest', seed=seed)
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
    # y_true = tf.cast(y_true, tf.float32)
    dice_total = 0

    for idx in range(0, config.CLASSES):  # set first argument to 1 to ignore 0 class
        mask = tf.cast(tf.equal(y_true, idx), tf.float32)
        dice_total += dice_coef(mask, y_pred[..., idx])

    return 1 - (dice_total/(config.CLASSES))


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
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
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
        padding='same', activation='softmax')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def get_dataset(images_path, masks_path):
    images = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    masks = [os.path.join(masks_path, f) for f in os.listdir(masks_path)]

    datapoints_total = len(images)

    train_images = tf.data.Dataset.from_tensor_slices((images, masks)).map(__get_datapoint_images, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = tf.data.Dataset.from_tensor_slices((images, masks)).map(__get_datapoint_images, num_parallel_calls=tf.data.AUTOTUNE)

    return datapoints_total, train_images.map(__normalize), test_images.map(__normalize)


def __get_datapoint_images(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_bmp(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (128, 128))

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_bmp(mask)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, mask


def __normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


# def get_dataset(images_path, masks_path):
#     images = [cv2.imread(os.path.join(images_path, f), 1) for f in os.listdir(images_path)]
#     images = [tf.convert_to_tensor(f) for f in images]
#     images = [tf.image.resize(f, (128, 128)) for f in images]
#     datapoints_total = len(images)

#     masks = [cv2.imread(os.path.join(masks_path, f), 0) for f in os.listdir(masks_path)]
#     masks = [cv2.resize(f, (128, 128)) for f in masks]
#     masks = [np.expand_dims(f, axis=2) for f in masks]
#     masks = [tf.convert_to_tensor(f) for f in masks]

#     dataset = []

#     for idx, image in enumerate(images):
#         dataset.append(dict(
#             image=image,
#             segmentation_mask=masks[idx]
#         ))

#     train_images = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(dataset).to_dict(orient="list")).map(load_image)
#     test_images = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(dataset).to_dict(orient="list")).map(load_image)

#     return datapoints_total, train_images, test_images


# def normalize(input_image, input_mask):
#     input_image = tf.cast(input_image, tf.float32) / 255.0
#     return input_image, input_mask


# def load_image(datapoint):
#     input_image = datapoint['image']
#     input_mask = datapoint['segmentation_mask']

#     input_image, input_mask = normalize(input_image, input_mask)

#     return input_image, input_mask


def display(display_list, tensors=True):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        .map(Augment(seed=random.randint(1, 999)))
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size)

    print("Train Dataset:", train_batches)
    print("Val Dataset:", test_batches)

    model = get_model_p2pUnet(output_channels=output_classes)
    model.compile(optimizer='nadam',
                loss=dice_coef_loss,                 
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    _ = model.fit(train_batches, epochs=epochs,
                          steps_per_epoch=int(datapoints_total/batch_size),
                          validation_steps=int(datapoints_total/batch_size),
                          validation_data=test_batches)
    
    show_predictions(model, test_batches, 5)

    model.save(model_path)


def train_current_model(model_path, epochs, batch_size=64):
    
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

    datapoints_total, train_images, test_images = get_dataset(images_path, masks_path)

    train_batches = (
        train_images
        .cache()
        .shuffle(batch_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=random.randint(1, 999)))
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(batch_size)

    print("Train Dataset:", train_batches)
    print("Val Dataset:", test_batches)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam',
                loss=dice_coef_loss,                 
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    _ = model.fit(train_batches, epochs=epochs,
                          steps_per_epoch=int(datapoints_total/batch_size),
                          validation_steps=int(datapoints_total/batch_size),
                          validation_data=test_batches)
    
    show_predictions(model, test_batches, 5)

    model.save(model_path)
