import numpy as np
import cv2


def get_contours(binary_mask, method=cv2.RETR_TREE):
    mask = binary_mask * 255
    return cv2.findContours(mask.astype(np.uint8), method, cv2.CHAIN_APPROX_NONE)[-2]


def smooth_contours(contours, shape=(512, 512), kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))):
    if not contours:
        return contours

    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    blur = cv2.GaussianBlur(mask, (15,15), 0)
    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[-1]

    contours = get_contours(thresh)
    return contours
