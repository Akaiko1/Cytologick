import tensorflow as tf
import cv2

from demetra import ai, inference


def main():
    model = tf.keras.models.load_model('demetra.h5')
    model.save('demetra_new')


if __name__ == '__main__':
    main()
