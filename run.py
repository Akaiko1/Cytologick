import tensorflow as tf
import numpy as np
import cv2

import ai_tools
import config

def image_to_tensor(image):
    image = tf.convert_to_tensor(image/255.0)
    image = tf.image.resize(image, (128, 128)) 
    image = tf.expand_dims(image, axis=0)
    return image

test_image = cv2.imread('test.bmp', 1)
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

pads = int(test_image.shape[0]%config.IMAGE_SHAPE[0]), int(test_image.shape[1]%config.IMAGE_SHAPE[1])
test_image = cv2.copyMakeBorder(test_image, 0, 128 - pads[0], 0, 128 - pads[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

result_image = np.zeros(test_image.shape[:2])

model = tf.keras.models.load_model('demetra')

for x in range(0, test_image.shape[0], 128):
    for y in range(0, test_image.shape[1], 128):
        patch = image_to_tensor(test_image[x: x + 128, y: y + 128])
        pred_mask = model.predict(patch)
        # ai_tools.display([patch[0], ai_tools.create_mask(pred_mask)])
        result_image[x: x + 128, y: y + 128] = ai_tools.create_mask(pred_mask)[..., 0]

ai_tools.display([test_image, result_image], tensors=False)
