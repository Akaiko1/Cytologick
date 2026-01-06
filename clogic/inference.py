"""
TensorFlow inference module (DEPRECATED).

This module is deprecated in favor of inference_pytorch.py.
It remains for backward compatibility with remote TensorFlow Serving endpoints.
For local inference, use inference_pytorch.py instead.
"""

import warnings
warnings.warn(
    "clogic.inference (TensorFlow) is deprecated. Use clogic.inference_pytorch instead.",
    DeprecationWarning,
    stacklevel=2
)

import tensorflow as tf
import numpy as np
import cv2

import clogic.ai as ai
import tfs_connector as tfs
import clogic.smooth as smooth
from config import Config, load_config


def _resolve_cfg(cfg: Config | None) -> Config:
    return cfg if cfg is not None else load_config()


def image_to_tensor(image, add_dim: bool = True, cfg: Config | None = None, shapes=None):
    cfg = _resolve_cfg(cfg)
    if shapes is None:
        shapes = tuple(getattr(cfg, 'IMAGE_SHAPE', (128, 128)))

    image = tf.convert_to_tensor(image/255.0)
    image = tf.image.resize(image, shapes)

    if add_dim:
        image = tf.expand_dims(image, axis=0)

    return image


def apply_model(source, model, shapes=None, cfg: Config | None = None):
    cfg = _resolve_cfg(cfg)
    if shapes is None:
        shapes = tuple(getattr(cfg, 'IMAGE_SHAPE', (128, 128)))

    pad_h = (shapes[0] - (source.shape[0] % shapes[0])) % shapes[0]
    pad_w = (shapes[1] - (source.shape[1] % shapes[1])) % shapes[1]
    source_pads = cv2.copyMakeBorder(source, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    pathology_map = np.zeros(source_pads.shape[:2])

    for x in range(0, source_pads.shape[0], shapes[0]):
        for y in range(0, source_pads.shape[1], shapes[1]):
            patch = source_pads[x: x + shapes[0], y: y + shapes[1]]
            pred_mask = model(image_to_tensor(patch, cfg=cfg, shapes=shapes))
            prediction = np.asarray(ai.create_mask(pred_mask)[..., 0])
            pathology_map[x: x + shapes[0], y: y + shapes[1]] = cv2.resize(prediction.astype(np.uint8), shapes)

    return pathology_map[:source.shape[0], :source.shape[1]]


def apply_model_raw(source, model, classes, shapes=None, cfg: Config | None = None):
    cfg = _resolve_cfg(cfg)
    if shapes is None:
        shapes = tuple(getattr(cfg, 'IMAGE_SHAPE', (128, 128)))

    pad_h = (shapes[0] - (source.shape[0] % shapes[0])) % shapes[0]
    pad_w = (shapes[1] - (source.shape[1] % shapes[1])) % shapes[1]
    source_pads = cv2.copyMakeBorder(source, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    pathology_map = np.zeros((source_pads.shape[0], source_pads.shape[1], classes))

    for x in range(0, source_pads.shape[0], shapes[0]):
        for y in range(0, source_pads.shape[1], shapes[1]):
            patch = source_pads[x: x + shapes[0], y: y + shapes[1]]
            pred_mask = model(image_to_tensor(patch, cfg=cfg, shapes=shapes))
            prediction = np.asarray(pred_mask)[0]
            pathology_map[x: x + shapes[0], y: y + shapes[1]] = cv2.resize(prediction, shapes)

    return pathology_map[:source.shape[0], :source.shape[1]]


def apply_remote(source, chunk_size=(256, 256), model_input_size=(128, 128), endpoint_url='http://51.250.28.160:7500',
                  model_name='demetra', batch_size=2, parallelism_mode=1, thread_count=4):
    """Aplly cloud model"""
    resize_ops = tfs.ResizeOptions(chunk_size=chunk_size, model_input_size=model_input_size)
    pathology_map = tfs.apply_segmentation_model_parallel([source], endpoint_url=endpoint_url, model_name=model_name, batch_size=batch_size,
                                                  resize_options=resize_ops, normalization=lambda x: x/255, parallelism_mode=parallelism_mode, thread_count=thread_count)

    return pathology_map[0][:source.shape[0], :source.shape[1]]


def apply_model_smooth(source, model, shape=None, cfg: Config | None = None, classes: int | None = None):
        cfg = _resolve_cfg(cfg)
        if shape is None:
            shape = int(tuple(getattr(cfg, 'IMAGE_SHAPE', (128, 128)))[0])
        if classes is None:
            classes = int(getattr(cfg, 'CLASSES', 3))
           
        predictions_smooth = smooth.predict_img_with_smooth_windowing(
            source,
            window_size=shape,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=classes,
            pred_func=(
                lambda img_batch_subdiv: model(image_to_tensor(img_batch_subdiv, add_dim=False, cfg=cfg, shapes=(shape, shape)))
            )
        )
           
        return np.argmax(predictions_smooth, axis=2)
