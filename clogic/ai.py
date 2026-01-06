"""clogic.ai (TensorFlow) (DEPRECATED).

This module contains the legacy TensorFlow/Keras training pipeline.
It is deprecated in favor of the PyTorch pipeline.
"""

import warnings

warnings.warn(
    "clogic.ai (TensorFlow/Keras training pipeline) is deprecated. Prefer the PyTorch pipeline.",
    DeprecationWarning,
    stacklevel=2,
)

import os
import random
from typing import Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import config

# Configure segmentation models framework
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm
sm.set_framework('tf.keras')

# Configure Keras backend
keras.backend.set_image_data_format('channels_last')

# Suppress KMP duplicate library warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global configurations
KERAS_BACKEND = tf.keras.backend
MODEL_CALLBACKS = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='demetra_checkpoint.keras',
        monitor='val_iou_score',
        mode='max',
        save_best_only=True
    )
]


class Augment(tf.keras.layers.Layer):
    """
    Data augmentation layer for medical image segmentation.
    
    Applies synchronized augmentations to both input images and corresponding
    segmentation masks to ensure consistency during training.
    """

    def __init__(self, seed: int = 34):
        """
        Initialize the augmentation layer.
        
        Args:
            seed: Random seed for reproducible augmentations
        """
        super().__init__()
        
        # Input image augmentations (includes color/noise transforms)
        self.augment_inputs = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(1.0, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomTranslation(0.35, 0.35, fill_mode="reflect", seed=seed),
            tf.keras.layers.GaussianNoise(0.25, seed=seed),
            tf.keras.layers.RandomZoom(0.2, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomContrast(0.25, seed=seed)
        ])

        # Mask augmentations (geometric transforms only, nearest interpolation)
        self.augment_labels = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(1.0, fill_mode="reflect", interpolation='nearest', seed=seed),
            tf.keras.layers.RandomTranslation(0.35, 0.35, fill_mode="reflect", interpolation='nearest', seed=seed),
            tf.keras.layers.RandomZoom(0.2, fill_mode="reflect", interpolation='nearest', seed=seed)
        ])

    def call(self, inputs, labels):
        """
        Apply augmentations to inputs and labels.
        
        Args:
            inputs: Input images tensor
            labels: Segmentation masks tensor
            
        Returns:
            Tuple of augmented (inputs, labels)
        """
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        labels = tf.cast(labels, tf.uint8)

        return inputs, labels


def get_dataset(images_path: str, masks_path: str) -> Tuple[int, tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets from image and mask directories.
    
    Args:
        images_path: Path to directory containing training images
        masks_path: Path to directory containing corresponding masks
        
    Returns:
        Tuple containing (total_datapoints, train_dataset, test_dataset)
    """
    images = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    masks = [os.path.join(masks_path, f) for f in os.listdir(masks_path)]

    datapoints_total = len(images)

    train_images = tf.data.Dataset.from_tensor_slices((images, masks)).map(
        _load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_images = tf.data.Dataset.from_tensor_slices((images, masks)).map(
        _load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )

    return datapoints_total, train_images, test_images


def _load_and_preprocess_image(image_path, mask_path):
    """
    Load and preprocess a single image-mask pair.
    
    Args:
        image_path: Path to the input image file
        mask_path: Path to the corresponding mask file
        
    Returns:
        Tuple of (preprocessed_image, preprocessed_mask)
    """
    # Load and decode image
    image = tf.io.read_file(image_path)
    image = tf.io.decode_bmp(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=(128, 128))

    # Load and decode mask
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_bmp(mask)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(images=mask, size=(128, 128), method='nearest')
    
    # Normalize image values to [0, 1]
    image, mask = _normalize_image(image, mask)

    return image, mask


def _normalize_image(image, mask):
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image tensor
        mask: Input mask tensor
        
    Returns:
        Tuple of (normalized_image, mask)
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, mask


def _convert_to_categorical(image, mask):
    """
    Convert mask to categorical (one-hot) format.
    
    Args:
        image: Input image tensor
        mask: Input mask tensor
        
    Returns:
        Tuple of (image, categorical_mask)
    """
    mask = tf.squeeze(mask, axis=-1) 
    mask = tf.one_hot(tf.cast(mask, tf.int32), 3)
    mask = tf.cast(mask, tf.float32)
    return image, mask


def display(display_list, tensors: bool = True):
    """
    Display a list of images in a single row for comparison.
    
    Args:
        display_list: List of images to display
        tensors: Whether the images are TensorFlow tensors (default: True)
    """
    plt.figure(figsize=(15, 15))

    titles = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titles[i])
        if tensors:
            plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        else:
            plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    """
    Create a mask from model predictions by taking the argmax.
    
    Args:
        pred_mask: Model prediction tensor
        
    Returns:
        Predicted mask tensor
    """
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, dataset=None, num: int = 1):
    """
    Display model predictions on sample images from dataset.
    
    Args:
        model: Trained model for inference
        dataset: Dataset to sample from (optional)
        num: Number of samples to display (default: 1)
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])


def add_sample_weights(image, label):
    """
    Add sample weights to give higher importance to certain classes.
    
    Args:
        image: Input image tensor
        label: Label tensor
        
    Returns:
        Tuple of (image, label, sample_weights)
    """
    # Define class weights (giving more weight to pathological findings)
    class_weights = tf.constant([1.0, 1.0, 2.0])
    class_weights = class_weights / tf.reduce_sum(class_weights)
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


def train_new_model(model_path: str, output_classes: int, epochs: int, batch_size: int = 64):
    """
    Train a new U-Net model with EfficientNetB3 encoder for medical image segmentation.

    Creates and trains a new segmentation model from scratch using the provided dataset.
    The model uses transfer learning with ImageNet pretrained weights and applies
    data augmentation for improved generalization.

    Args:
        model_path: Path where the trained model will be saved
        output_classes: Number of segmentation classes (e.g., 3 for background, LSIL, HSIL)
        epochs: Number of training epochs
        batch_size: Training batch size (default: 64)

    Notes:
        - Uses EfficientNetB3 as encoder with ImageNet pretrained weights
        - Applies comprehensive data augmentation to training data
        - Combines Jaccard loss and Categorical Focal loss for training
        - Monitors IoU score and F-score metrics during training
        - Datasets are loaded from config.DATASET_FOLDER structure
    """
    # Construct dataset paths from configuration
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

    # Load and prepare datasets
    datapoints_total, train_images, test_images = get_dataset(images_path, masks_path)

    # Configure training pipeline with augmentation
    train_batches = (
        train_images
        .cache()
        .shuffle(batch_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=random.randint(1, 999)))
        .map(_convert_to_categorical)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Configure validation pipeline (no augmentation)
    test_batches = test_images.batch(batch_size).map(_convert_to_categorical)
    
    print(f"Train Dataset: {train_batches}")
    print(f"Validation Dataset: {test_batches}")

    # Create U-Net model with EfficientNetB3 backbone
    model = sm.Unet(
        'efficientnetb3',
        classes=output_classes,
        encoder_weights='imagenet',
        activation='softmax'
    )
    
    # Compile model with combined loss and metrics
    model.compile(
        optimizer='nadam',
        loss=sm.losses.JaccardLoss() + sm.losses.CategoricalFocalLoss(),
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()]
    )

    # Train the model
    training_history = model.fit(
        train_batches,
        epochs=epochs,
        steps_per_epoch=int(datapoints_total / batch_size),
        validation_steps=int(datapoints_total / batch_size),
        validation_data=test_batches,
        callbacks=[]
    )

    # Save the trained model
    model.save(model_path)


def train_current_model(model_path: str, epochs: int, batch_size: int = 64, lr: float = 0.0001):
    """
    Continue training an existing pre-trained model with fine-tuning.

    Loads a previously trained model and continues training with new data or
    additional epochs. This function is useful for transfer learning or when
    resuming interrupted training sessions.

    Args:
        model_path: Path to the existing trained model to load and continue training
        epochs: Number of additional epochs to train
        batch_size: Training batch size (default: 64)
        lr: Learning rate for the Adam optimizer (default: 0.0001)

    Notes:
        - Loads existing model without compilation to allow custom optimizer settings
        - Uses Adam optimizer with configurable learning rate
        - Applies same loss combination as new model training
        - Includes model checkpointing to save best weights during training
        - Datasets are loaded from config.DATASET_FOLDER structure
    """
    # Construct dataset paths from configuration
    images_path = os.path.join(config.DATASET_FOLDER, config.IMAGES_FOLDER)
    masks_path = os.path.join(config.DATASET_FOLDER, config.MASKS_FOLDER)

    # Load and prepare datasets
    datapoints_total, train_images, test_images = get_dataset(images_path, masks_path)

    # Configure training pipeline with augmentation
    train_batches = (
        train_images
        .cache()
        .shuffle(batch_size)
        .batch(batch_size)
        .repeat()
        .map(Augment(seed=random.randint(1, 999)))
        .map(_convert_to_categorical)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Configure validation pipeline (no augmentation)
    test_batches = test_images.batch(batch_size).map(_convert_to_categorical)
    
    print(f"Train Dataset: {train_batches}")
    print(f"Validation Dataset: {test_batches}")

    # Load existing model and recompile with new optimizer settings
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=sm.losses.JaccardLoss() + sm.losses.CategoricalFocalLoss(),
        metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()]
    )

    # Continue training with checkpointing
    training_history = model.fit(
        train_batches,
        epochs=epochs,
        steps_per_epoch=int(datapoints_total / batch_size),
        validation_steps=int(datapoints_total / batch_size),
        validation_data=test_batches,
        callbacks=MODEL_CALLBACKS
    )

    # Save the updated model
    model.save(model_path)
