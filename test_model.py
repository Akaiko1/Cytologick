import tensorflow as tf
import cv2

from demetra import ai, inference

def main():
    source = cv2.imread('test.bmp', 1)
    source = cv2.resize(source, (int(source.shape[1]/2), int(source.shape[0]/2)))

    model = tf.keras.models.load_model('demetra_main')
    pathology_map = inference.apply_model(source, model)

    ai.display([source, pathology_map], tensors=False)



if __name__ == '__main__':
    main()