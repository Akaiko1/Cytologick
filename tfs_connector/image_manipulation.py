from typing import Optional, Tuple, List, Callable

import cv2
import numpy as np

from tfs_connector.metrics import log_metrics


def pad_image(image: np.ndarray, chunk_size: Optional[Tuple[int, int]] = (128, 128)) -> np.ndarray:
    """
    Pads image for it to be divided evenly into chunks of given size
    :param image: Original image
    :param chunk_size: Size of a chunk
    :return: Padded image
    """
    pads = int(image.shape[0] % chunk_size[0]), int(image.shape[1] % chunk_size[1])

    if pads == (0, 0):
        return image

    source_pads = cv2.copyMakeBorder(image, top=0,
                                     bottom=chunk_size[1] - pads[0],
                                     left=0,
                                     right=chunk_size[0] - pads[1],
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=(255, 255, 255))

    return source_pads


def resize_image(image: np.ndarray, size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """
    Resizes image to a given size. If an image is already given size no modifications will be made.
    :param image: Source image
    :param size: Target size
    :param interpolation: cv2 interpolation algorythm
    :return:
    """
    if not image.shape[0] == size:
        return cv2.resize(image, size, interpolation=interpolation)
    return image


def image_to_tensor(image: np.ndarray,
                    normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """
    Converts an image into a float tensor
    :param image: image to convert into a float tensor
    :param normalization: a normalization algorythm function
    """

    image_as_tensor = normalization(image) if normalization is not None else image
    return image_as_tensor


def __get_new_size(shape: Tuple[int, int], ann_metadata_size: Tuple[int, int]):
    """
    Calculate the size of a patch that ANN waits for. The main problem in that shape of ndarray is reversed: it's (y, x)
    instead of (x, y), while ANN metadata gives usual (x, y) dimensions
    :param shape: shape of original image
    :param ann_metadata_size: size of ANN input as metadata it gives. -1 stands for "any length"
    :return: New size to resize to
    """
    if ann_metadata_size == (-1, -1):
        return shape[1], shape[0]

    if -1 not in ann_metadata_size:
        return ann_metadata_size

    if ann_metadata_size[0] == -1:
        return int(ann_metadata_size[1] * shape[1] / shape[0]), ann_metadata_size[1]

    if ann_metadata_size[1] == -1:
        return ann_metadata_size[0], int(ann_metadata_size[0] * shape[0] / shape[1])


def get_image_patch(image: np.ndarray, patch_size: Tuple[int, int], patch_position: Tuple[int, int],
                    model_input_size: Tuple[int, int], interpolation: int) -> np.ndarray:
    """
    Gets a patch of an image based on a chunk size and chunk position and resizes it to a new size
    :param interpolation: Interpolation algorythm used to resize images. Use cv2.INTER_* constants.
    :param image: Image we get a probe from
    :param patch_size: Size of a patch
    :param patch_position: Position of a patch (in a grid of patches)
    :param model_input_size: Size of ANN input
    :return: Patch of an image
    """
    pads_patch: np.ndarray = image[
                             patch_position[0]: patch_position[0] + patch_size[0],
                             patch_position[1]: patch_position[1] + patch_size[1]
                             ]

    resize_to = __get_new_size(pads_patch.shape, model_input_size)

    resized_pads_patch = resize_image(pads_patch, resize_to, interpolation)
    return resized_pads_patch


def cut_image_into_chunks(image: np.ndarray, chunk_size: Tuple[int, int],
                          model_input_size: Tuple[int, int],
                          interpolation: int,
                          normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> List[np.ndarray]:
    """
    Cuts (padded) image into a chunks of given size, resized into a model_input_size
    :param interpolation: Interpolation algorythm used to resize images. Use cv2.INTER_* constants.
    :param normalization: a normalization algorythm function
    :param image: Image to be cut into chunks
    :param chunk_size: Size of a chunk
    :param model_input_size: Size of a model input
    :return: Linear list of tensors sized model_input_size
    """
    result: List[np.ndarray] = []
    for x in range(0, image.shape[0], chunk_size[0]):
        for y in range(0, image.shape[1], chunk_size[1]):
            patch = get_image_patch(image, chunk_size, (x, y), model_input_size,
                                    interpolation=interpolation)
            input_tensor = image_to_tensor(patch, normalization=normalization)
            result.append(input_tensor)
    return result


@log_metrics('irym_tfs_connector.metrics')
def pad_images(images: List[np.ndarray], chunk_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Pads a list of images according to a chunk_size. Does nothing if images are divisible into chunks without
    extra padding.
    :param images: A list of images to pad
    :param chunk_size: Size of a chunk resulting images will be brought to by padding
    :return: A list of padded images
    """
    padding = int(images[0].shape[0] % chunk_size[0]), int(images[0].shape[1] % chunk_size[1])
    if padding != (0, 0):
        images = [pad_image(image, chunk_size) for image in images]
    return images
