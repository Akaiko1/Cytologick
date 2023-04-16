import tensorflow as tf
import numpy as np
import cv2

import demetra.ai as ai
import demetra.smooth as smooth
import config


def image_to_tensor(image, add_dim=True):
    image = tf.convert_to_tensor(image/255.0)
    image = tf.image.resize(image, config.IMAGE_SHAPE)

    if add_dim:
        image = tf.expand_dims(image, axis=0)

    return image


def apply_model(source, model, shapes=config.IMAGE_SHAPE):
    pads = int(source.shape[0]%shapes[0]), int(source.shape[1]%shapes[1])
    source_pads = cv2.copyMakeBorder(source, 0, shapes[0] - pads[0], 0, shapes[1] - pads[1], cv2.BORDER_REPLICATE)

    pathology_map = np.zeros(source_pads.shape[:2])

    for x in range(0, source_pads.shape[0], shapes[0]):
        for y in range(0, source_pads.shape[1], shapes[1]):
            patch = source_pads[x: x + shapes[0], y: y + shapes[1]]
            pred_mask = model(image_to_tensor(patch))
            prediction = np.asarray(ai.create_mask(pred_mask)[..., 0])
            pathology_map[x: x + shapes[0], y: y + shapes[1]] = cv2.resize(prediction.astype(np.uint8), shapes)

    return pathology_map[:source.shape[0], :source.shape[1]]


def apply_model_smooth(source, model, shape=config.IMAGE_SHAPE[0]):
           
        predictions_smooth = smooth.predict_img_with_smooth_windowing(
            source,
            window_size=shape,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=config.CLASSES,
            pred_func=(
                lambda img_batch_subdiv: model(image_to_tensor(img_batch_subdiv, add_dim=False))
            )
        )
           
        return np.argmax(predictions_smooth, axis=2)
