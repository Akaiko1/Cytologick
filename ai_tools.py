from tensorflow_examples.models.pix2pix import pix2pix
from pprint import pprint

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import config
import cv2
import os


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


def unet_model(output_channels:int):
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
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


def load_image(datapoint):
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def display(display_list):
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, dataset=None, num=2):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])


def main():
    # import tensorflow_datasets as tfds
    # ddataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    dataset = []

    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

    images = [cv2.imread(os.path.join(images_path, f), 1) for f in os.listdir(images_path)]
    masks = [cv2.imread(os.path.join(masks_path, f), 0) for f in os.listdir(masks_path)]
    masks = [cv2.resize(f, (128, 128)) for f in masks]
    masks = [np.expand_dims(f, axis=2) for f in masks]

    images = [tf.convert_to_tensor(f) for f in images]
    masks = [tf.convert_to_tensor(f) for f in masks]

    images = [tf.image.resize(f, (128, 128)) for f in images]

    for idx, image in enumerate(images):
        dataset.append(dict(
            image=image,
            segmentation_mask=masks[idx]
        ))

    # for entry in ddataset['train']:
    #     load_image(entry)

    train_images = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(dataset[:10]).to_dict(orient="list")).map(load_image)
    test_images = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(dataset[10:]).to_dict(orient="list")).map(load_image)


    train_batches = (
        train_images
        .cache()
        .shuffle(3)
        .batch(3)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(3)

    # for images, masks in train_batches.take(2):
    #     sample_image, sample_mask = images[0], masks[0]
    #     display([sample_image, sample_mask])
    
    OUTPUT_CLASSES = 2

    model = unet_model(output_channels=OUTPUT_CLASSES)
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    show_predictions(model, train_batches)

    EPOCHS = 200
    VALIDATION_STEPS = 5

    model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=5,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches)
    
    show_predictions(model, test_batches)

if __name__ == '__main__':
    main()
