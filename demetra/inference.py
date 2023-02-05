import tensorflow as tf
import numpy as np
import cv2

import demetra.ai as ai
import config


def image_to_tensor(image):
    image = tf.convert_to_tensor(image/255.0)
    image = tf.image.resize(image, (128, 128)) 
    image = tf.expand_dims(image, axis=0)
    return image


def apply_model(source, model):
    pads = int(source.shape[0]%config.IMAGE_SHAPE[0]), int(source.shape[1]%config.IMAGE_SHAPE[1])
    source_pads = cv2.copyMakeBorder(source, 0, 128 - pads[0], 0, 128 - pads[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

    pathology_map = np.zeros(source_pads.shape[:2])

    for x in range(0, source_pads.shape[0], 128):
        for y in range(0, source_pads.shape[1], 128):
            patch = image_to_tensor(source_pads[x: x + 128, y: y + 128])
            pred_mask = model(patch)
            pathology_map[x: x + 128, y: y + 128] = ai.create_mask(pred_mask)[..., 0]

    return pathology_map[:source.shape[0], :source.shape[1]]
