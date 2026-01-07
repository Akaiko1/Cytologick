from __future__ import annotations

import configparser
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import yaml


def _abs_path(path: str) -> str:
    return os.path.abspath(path)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {'1', 'true', 'yes', 'y', 'on'}


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value).strip())


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def _to_tuple2(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_to_int(value[0]), _to_int(value[1]))
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace('x', ',').split(',') if p.strip()]
        if len(parts) == 2:
            return (_to_int(parts[0]), _to_int(parts[1]))
    raise ValueError(f'Expected 2-tuple, got: {value!r}')


@dataclass
class Config:
    # ---------------------------------------------------------------------
    # General
    # ---------------------------------------------------------------------
    CURRENT_SLIDE: str = field(default_factory=lambda: _abs_path('slide-2022-09-12T15-38-25-R1-S2.mrxs'))
    CURRENT_SLIDE_XML: str = field(default_factory=lambda: os.path.join('current', 'slide-2022-09-12T15-38-25-R1-S2', 'Data0021.dat'))
    OPENSLIDE_PATH: str = field(default_factory=lambda: _abs_path(os.path.join('openslide', 'bin')))
    HDD_SLIDES: str = field(default_factory=lambda: _abs_path('current'))
    HDD_SLIDES_SVS: str = 'E:\\CYTOLOGY_2'  # legacy Windows path
    TEMP_FOLDER: str = 'temp'

    # ---------------------------------------------------------------------
    # Neural network
    # ---------------------------------------------------------------------
    FRAMEWORK: str = 'pytorch'  # 'pytorch' (TensorFlow path is deprecated)
    DATASET_FOLDER: str = 'dataset'
    MASKS_FOLDER: str = 'masks'
    IMAGES_FOLDER: str = 'rois'
    IMAGE_CHUNK: tuple[int, int] = (256, 256)
    IMAGE_SHAPE: tuple[int, int] = (128, 128)
    CLASSES: int = 3

    PT_ENCODER_NAME: str = 'efficientnet-b3'
    PT_ENCODER_WEIGHTS: Any = 'imagenet'
    PT_USE_ENCODER_PREPROCESSING: bool = False

    PT_OPTIMIZER: str = 'adamw'
    PT_LR: float = 1e-3
    PT_WEIGHT_DECAY: float = 1e-4
    PT_ENCODER_LR_MULT: float = 0.1
    PT_GRAD_CLIP_NORM: float = 1.0
    # With mixup enabled in training loop, default to 0.0 to avoid over-regularization.
    PT_LABEL_SMOOTHING: float = 0.0
    # Mixup alpha for Beta(alpha, alpha). 0 disables mixup.
    PT_MIXUP_ALPHA: float = 0.2
    PT_NUM_WORKERS: int = -1  # -1 = auto; 0 disables multiprocessing DataLoader

    LABELS: dict[str, int] = field(
        default_factory=lambda: {
            'LSIL': 2,
            'HSIL': 2,
            'Group HSIL': 2,
            'ASCH': 2,
            'Group atypical': 2,
            'ASCUS': 2,
            'Atypical': 2,
            'Atypical naked': 2,
        }
    )

    # ---------------------------------------------------------------------
    # GUI
    # ---------------------------------------------------------------------
    SLIDE_DIR: str = field(default_factory=lambda: os.path.join('.', 'current'))
    UNET_PRED_MODE: str = 'remote'  # 'smooth', 'direct', 'remote'
    USE_TTA: bool = False

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------
    EXCLUDE_DUPLICATES: bool = False
    BROADEN_INDIVIDUAL_RECT: int = 1000

    # ---------------------------------------------------------------------
    # Web
    # ---------------------------------------------------------------------
    IP_EXPOSED: str = '127.0.0.1'
    ENDPOINT_URL: str = 'http://51.250.28.160:7500'
    HEALTH_TIMEOUT: float = 1.5
    WEB_CONF_THRESHOLD: float = 0.6
    WEB_ATYPICAL_CLASS_INDEX: int = 2
    WEB_FAST_TILES: bool = True
    WEB_TILE_CACHE_SIZE: int = 256
    WEB_SCAN_ALL_TISSUE: bool = False
    WEB_PROGRESS_EVERY: int = 25
    WEB_MIN_TISSUE_FRACTION: float = 0.05
    WEB_TISSUE_GRAY_THRESHOLD: int = 220
    WEB_PT_BATCH_SIZE: int = 0


def _load_ini_to_mapping(config_file: str) -> dict[str, dict[str, Any]]:
    parser = configparser.ConfigParser()
    parser.read(config_file)
    return {
        'general': dict(parser['General']) if 'General' in parser else {},
        'neural_network': dict(parser['Neural Network']) if 'Neural Network' in parser else {},
        'gui': dict(parser['GUI']) if 'GUI' in parser else {},
        'dataset': dict(parser['Dataset']) if 'Dataset' in parser else {},
        'web': dict(parser['Web']) if 'Web' in parser else {},
    }


def _apply_mapping(cfg: Config, data: Mapping[str, Any]) -> None:
    general = data.get('general', {}) or {}
    if 'current_slide' in general:
        cfg.CURRENT_SLIDE = _abs_path(str(general['current_slide']))
    if 'current_slide_xml' in general:
        cfg.CURRENT_SLIDE_XML = str(general['current_slide_xml'])
    if 'openslide_path' in general:
        cfg.OPENSLIDE_PATH = _abs_path(str(general['openslide_path']))
    if 'hdd_slides' in general:
        cfg.HDD_SLIDES = _abs_path(str(general['hdd_slides']))
    if 'hdd_slides_svs' in general:
        cfg.HDD_SLIDES_SVS = str(general['hdd_slides_svs'])
    if 'temp_folder' in general:
        cfg.TEMP_FOLDER = str(general['temp_folder'])

    neural = data.get('neural_network', {}) or {}
    if 'framework' in neural:
        cfg.FRAMEWORK = str(neural['framework'])
    if 'dataset_folder' in neural:
        cfg.DATASET_FOLDER = str(neural['dataset_folder'])
    if 'masks_folder' in neural:
        cfg.MASKS_FOLDER = str(neural['masks_folder'])
    if 'images_folder' in neural:
        cfg.IMAGES_FOLDER = str(neural['images_folder'])
    if 'image_chunk' in neural:
        cfg.IMAGE_CHUNK = _to_tuple2(neural['image_chunk'])
    if 'image_shape' in neural:
        cfg.IMAGE_SHAPE = _to_tuple2(neural['image_shape'])
    if 'classes' in neural:
        cfg.CLASSES = _to_int(neural['classes'])

    if 'pt_encoder_name' in neural:
        cfg.PT_ENCODER_NAME = str(neural['pt_encoder_name'])
    if 'pt_encoder_weights' in neural:
        cfg.PT_ENCODER_WEIGHTS = neural['pt_encoder_weights']
    if 'pt_use_encoder_preprocessing' in neural:
        cfg.PT_USE_ENCODER_PREPROCESSING = _to_bool(neural['pt_use_encoder_preprocessing'])

    if 'pt_optimizer' in neural:
        cfg.PT_OPTIMIZER = str(neural['pt_optimizer']).lower()
    if 'pt_lr' in neural:
        cfg.PT_LR = _to_float(neural['pt_lr'])
    if 'pt_weight_decay' in neural:
        cfg.PT_WEIGHT_DECAY = _to_float(neural['pt_weight_decay'])
    if 'pt_encoder_lr_mult' in neural:
        cfg.PT_ENCODER_LR_MULT = _to_float(neural['pt_encoder_lr_mult'])
    if 'pt_grad_clip_norm' in neural:
        cfg.PT_GRAD_CLIP_NORM = _to_float(neural['pt_grad_clip_norm'])

    if 'pt_label_smoothing' in neural:
        cfg.PT_LABEL_SMOOTHING = _to_float(neural['pt_label_smoothing'])

    if 'pt_mixup_alpha' in neural:
        cfg.PT_MIXUP_ALPHA = _to_float(neural['pt_mixup_alpha'])

    if 'pt_num_workers' in neural:
        cfg.PT_NUM_WORKERS = _to_int(neural['pt_num_workers'])

    if 'labels' in neural and neural['labels'] is not None:
        cfg.LABELS = dict(neural['labels'])

    gui = data.get('gui', {}) or {}
    if 'slide_dir' in gui:
        cfg.SLIDE_DIR = str(gui['slide_dir'])
    if 'unet_pred_mode' in gui:
        cfg.UNET_PRED_MODE = str(gui['unet_pred_mode'])
    if 'use_tta' in gui:
        cfg.USE_TTA = _to_bool(gui['use_tta'])

    dataset = data.get('dataset', {}) or {}
    if 'exclude_duplicates' in dataset:
        cfg.EXCLUDE_DUPLICATES = _to_bool(dataset['exclude_duplicates'])
    if 'broaden_individual_rect' in dataset:
        cfg.BROADEN_INDIVIDUAL_RECT = _to_int(dataset['broaden_individual_rect'])

    web = data.get('web', {}) or {}
    if 'ip_exposed' in web:
        cfg.IP_EXPOSED = str(web['ip_exposed'])
    if 'endpoint_url' in web:
        cfg.ENDPOINT_URL = str(web['endpoint_url'])
    if 'health_timeout' in web:
        cfg.HEALTH_TIMEOUT = _to_float(web['health_timeout'])
    if 'web_conf_threshold' in web:
        cfg.WEB_CONF_THRESHOLD = _to_float(web['web_conf_threshold'])
    if 'atypical_class_index' in web:
        cfg.WEB_ATYPICAL_CLASS_INDEX = _to_int(web['atypical_class_index'])
    if 'fast_tiles' in web:
        cfg.WEB_FAST_TILES = _to_bool(web['fast_tiles'])
    if 'tile_cache_size' in web:
        cfg.WEB_TILE_CACHE_SIZE = _to_int(web['tile_cache_size'])
    if 'scan_all_tissue' in web:
        cfg.WEB_SCAN_ALL_TISSUE = _to_bool(web['scan_all_tissue'])
    if 'progress_every' in web:
        cfg.WEB_PROGRESS_EVERY = _to_int(web['progress_every'])
    if 'min_tissue_fraction' in web:
        cfg.WEB_MIN_TISSUE_FRACTION = _to_float(web['min_tissue_fraction'])
    if 'tissue_gray_threshold' in web:
        cfg.WEB_TISSUE_GRAY_THRESHOLD = _to_int(web['tissue_gray_threshold'])
    if 'pt_batch_size' in web:
        cfg.WEB_PT_BATCH_SIZE = _to_int(web['pt_batch_size'])


def _find_default_config_file() -> Optional[str]:
    candidates: list[str] = []

    # 1) Current working directory (developer-friendly)
    for name in ('config.yaml', 'config.yml', 'config.ini'):
        candidates.append(os.path.join(os.getcwd(), name))

    # 2) Directory of the executable (PyInstaller / end-user friendly)
    exe_dir = os.path.dirname(os.path.abspath(getattr(sys, 'executable', '') or ''))
    if exe_dir:
        for name in ('config.yaml', 'config.yml', 'config.ini'):
            candidates.append(os.path.join(exe_dir, name))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration into a Config object.

    This function has no side-effects besides an optional console print.
    """
    cfg = Config()

    path = config_file or _find_default_config_file()
    if not path:
        return cfg

    if multiprocessing.current_process().name == 'MainProcess':
        print(f'{path} detected')

    if path.endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    elif path.endswith('.ini'):
        data = _load_ini_to_mapping(path)
    else:
        return cfg

    _apply_mapping(cfg, data)
    return cfg
