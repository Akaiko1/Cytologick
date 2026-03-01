from __future__ import annotations

import configparser
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import yaml

DEFAULT_IMAGE_SIZE: tuple[int, int] = (256, 256)
DEFAULT_IMAGE_SHAPE: tuple[int, int] = DEFAULT_IMAGE_SIZE
DEFAULT_IMAGE_CHUNK: tuple[int, int] = DEFAULT_IMAGE_SIZE


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


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_to_float(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parts = [p.strip() for p in text.split(',') if p.strip()]
        return [_to_float(p) for p in parts]
    raise ValueError(f'Expected list[float] or comma-separated string, got: {value!r}')


def _to_tuple2(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_to_int(value[0]), _to_int(value[1]))
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace('x', ',').split(',') if p.strip()]
        if len(parts) == 2:
            return (_to_int(parts[0]), _to_int(parts[1]))
    raise ValueError(f'Expected 2-tuple, got: {value!r}')


def get_image_shape(cfg: Optional[Any] = None) -> tuple[int, int]:
    value = getattr(cfg, 'IMAGE_SHAPE', None) if cfg is not None else None
    if value is None:
        return DEFAULT_IMAGE_SHAPE
    try:
        return _to_tuple2(value)
    except Exception:
        return DEFAULT_IMAGE_SHAPE


def get_image_chunk(cfg: Optional[Any] = None) -> tuple[int, int]:
    value = getattr(cfg, 'IMAGE_CHUNK', None) if cfg is not None else None
    if value is None:
        return DEFAULT_IMAGE_CHUNK
    try:
        return _to_tuple2(value)
    except Exception:
        return DEFAULT_IMAGE_CHUNK


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
    IMAGE_CHUNK: tuple[int, int] = DEFAULT_IMAGE_CHUNK
    IMAGE_SHAPE: tuple[int, int] = DEFAULT_IMAGE_SHAPE
    CLASSES: int = 3

    # Segmentation architecture:
    # - 'unetplusplus'
    # - 'deeplabv3plus'
    # - 'segformer'
    PT_MODEL_ARCH: str = 'segformer'
    # Encoder backbone depends on PT_MODEL_ARCH, e.g.:
    # - segformer: mit_b2
    # - deeplabv3plus: resnet50 or mit_b2
    # - unetplusplus: resnet34 or resnet50
    PT_ENCODER_NAME: str = 'mit_b2'
    PT_ENCODER_WEIGHTS: Any = 'imagenet'
    PT_USE_ENCODER_PREPROCESSING: bool = False

    PT_OPTIMIZER: str = 'adamw'
    # LR scheduler:
    # - 'cawr': CosineAnnealingWarmRestarts
    # - 'onecycle': OneCycleLR (stepped per-batch)
    # Legacy/global scheduler fallback used when mode-specific values are absent.
    PT_SCHEDULER: str = 'onecycle'
    # Scheduler for training from scratch (model_new.py).
    PT_SCHEDULER_NEW: str = 'onecycle'
    # Scheduler for resume/finetune (model_train.py).
    PT_SCHEDULER_FINETUNE: str = 'cawr'
    PT_ONECYCLE_PCT_START: float = 0.1
    # Global LR fallback. Prefer architecture-specific LR defaults below.
    PT_LR: float = 1e-3
    # Use architecture-specific learning rates by default.
    PT_USE_ARCH_LR_DEFAULTS: bool = True
    # Architecture LR defaults (tuned for this project setup; see config comments).
    PT_LR_UNETPLUSPLUS: float = 1e-3
    PT_LR_DEEPLABV3PLUS: float = 1e-3
    PT_LR_SEGFORMER: float = 2.4e-4
    PT_WEIGHT_DECAY: float = 1e-4
    PT_ENCODER_LR_MULT: float = 0.1
    PT_GRAD_CLIP_NORM: float = 1.0
    # Segmentation loss:
    # - 'combined': Lovasz + CrossEntropy
    # - 'focal_tversky_ce': Focal-Tversky + weighted CrossEntropy
    PT_LOSS: str = 'focal_tversky_ce'
    PT_LOSS_CE_WEIGHT: float = 0.5
    PT_LOSS_TVERSKY_WEIGHT: float = 1.0
    PT_TVERSKY_ALPHA: float = 0.3
    PT_TVERSKY_BETA: float = 0.7
    PT_TVERSKY_GAMMA: float = 1.33
    # Additional edge-aware loss term (0 disables).
    PT_BOUNDARY_LOSS_WEIGHT: float = 0.2
    PT_BOUNDARY_KERNEL_SIZE: int = 3
    PT_BOUNDARY_IGNORE_BACKGROUND: bool = True
    # Class weights for CE ordered as [background, normal, pathology].
    PT_CLASS_WEIGHTS: list[float] = field(default_factory=lambda: [0.25, 1.0, 2.5])
    # With mixup enabled in training loop, default to 0.0 to avoid over-regularization.
    PT_LABEL_SMOOTHING: float = 0.0
    # Mixup alpha for Beta(alpha, alpha). 0 disables mixup.
    PT_MIXUP_ALPHA: float = 0.2
    # Train loader sampling:
    # - 'none': standard shuffle
    # - 'pathology_balanced': weighted sampler to target pathology/non-pathology tile ratio
    PT_TRAIN_SAMPLER: str = 'pathology_balanced'
    PT_SAMPLER_TARGET_PATHOLOGY_FRACTION: float = 0.5
    PT_SAMPLER_PATHOLOGY_MIN_PIXELS: int = 10
    PT_SAMPLER_REPLACEMENT: bool = True
    # Stain-aware color strategy (HSV/HED + optional soft stain normalization).
    PT_STAIN_AWARE_AUG: bool = True
    PT_STAIN_HSV_PROB: float = 0.35
    PT_STAIN_HED_PROB: float = 0.25
    PT_STAIN_NORM_PROB: float = 0.20
    PT_STAIN_NORM_BLEND_RANGE: tuple[float, float] = (0.15, 0.35)
    PT_NUM_WORKERS: int = -1  # -1 = auto; 0 disables multiprocessing DataLoader
    # Save per-epoch checkpoints (*_epochNNN.pth). Disabled by default to reduce disk usage.
    PT_SAVE_EVERY_EPOCH: bool = False
    # Which validation metric to use for best-checkpoint selection.
    # Options: 'mean_iou', 'mean_f1', 'pathology_iou', 'pathology_f1', 'pathology_pr_auc'
    PT_CHECKPOINT_METRIC: str = 'pathology_iou'
    # Class index used as pathology/positive class for class-specific metrics.
    PT_PATHOLOGY_CLASS_INDEX: int = 2
    # Whether to include background class (0) in mean IoU/F1 reporting.
    PT_INCLUDE_BACKGROUND_METRICS: bool = False
    # Pathology augmentation safety: keep at least this fraction of pathology pixels.
    PT_PATHOLOGY_MIN_KEEP_RATIO: float = 0.9
    # Probability to skip all stochastic train augmentations for a sample.
    # Resize + preprocessing + tensor conversion are still applied.
    PT_AUG_SKIP_PROB: float = 0.2
    # Strict pathology augmentation mode: disable risky geometry/occlusion transforms.
    PT_PATHOLOGY_STRICT_AUG: bool = True
    # Max random rotation (degrees) for pathology-tiles conservative transform.
    PT_PATHOLOGY_ROTATE_LIMIT: int = 30

    # Validation split strategy
    # - 'cyclic': rotating/rolling holdout. Each epoch uses a different validation subset.
    # - 'static': a single random holdout subset used for all epochs.
    PT_VAL_STRATEGY: str = 'static'
    # Fraction of samples to use for validation (0..1). Default 0.2 -> 80/20.
    PT_VAL_FRACTION: float = 0.2
    # Seed for deterministic train/val split permutation.
    PT_VAL_SEED: int = 42

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
    # Tile overlap ratio (0.0 = no overlap, 0.5 = 50% overlap). Higher values
    # capture more cells at tile boundaries but increase dataset size.
    TILE_OVERLAP: float = 0.25
    # Use pathology-centered cropping instead of grid-based. Centers tiles on
    # pathology objects and adjusts size/position to avoid truncating them.
    # Generates both group tiles (multiple pathologies) and individual tiles.
    CENTERED_CROP: bool = True
    # Centered-crop algorithm:
    # - 'heuristic': shift-based search (fast, may miss valid placements)
    # - 'ring': exhaustive search for edge-safe tiles (slower, guarantees if exists)
    CENTERED_CROP_ALGO: str = 'heuristic'
    # Edge margin (pixels) to keep pathology away from tile borders.
    CENTERED_CROP_EDGE_MARGIN: int = 3
    # Dataset debug outputs (writes per-rect overlays to debug_info/).
    DATASET_DEBUG: bool = False
    # Pad each rect with a white border (pixels) before cropping.
    RECT_BORDER_PADDING: int = 0
    # Drop tiny pathology mask fragments (pixels) after rasterization.
    # Set to 0 to disable.
    MIN_PATHOLOGY_PIXELS: int = 10
    # Force extraction if individual crop fails by masking intruders.
    FORCE_EXTRACTION: bool = True
    # Normal tile limit for centered-crop:
    # - 'same': max(group+individual, 5)
    # - 'all': no limit (use all normal objects)
    # - 'multiplier': (group+individual) * NORMAL_TILES_MULTIPLIER
    NORMAL_TILES_LIMIT_MODE: str = 'same'
    NORMAL_TILES_MULTIPLIER: float = 1.0

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
    if 'pt_model_arch' in neural:
        cfg.PT_MODEL_ARCH = str(neural['pt_model_arch']).lower()
    if 'pt_encoder_weights' in neural:
        cfg.PT_ENCODER_WEIGHTS = neural['pt_encoder_weights']
    if 'pt_use_encoder_preprocessing' in neural:
        cfg.PT_USE_ENCODER_PREPROCESSING = _to_bool(neural['pt_use_encoder_preprocessing'])

    if 'pt_optimizer' in neural:
        cfg.PT_OPTIMIZER = str(neural['pt_optimizer']).lower()
    if 'pt_scheduler' in neural:
        cfg.PT_SCHEDULER = str(neural['pt_scheduler']).lower()
    if 'pt_scheduler_new' in neural:
        cfg.PT_SCHEDULER_NEW = str(neural['pt_scheduler_new']).lower()
    if 'pt_scheduler_finetune' in neural:
        cfg.PT_SCHEDULER_FINETUNE = str(neural['pt_scheduler_finetune']).lower()
    if 'pt_onecycle_pct_start' in neural:
        cfg.PT_ONECYCLE_PCT_START = _to_float(neural['pt_onecycle_pct_start'])
    if 'pt_lr' in neural:
        cfg.PT_LR = _to_float(neural['pt_lr'])
    if 'pt_use_arch_lr_defaults' in neural:
        cfg.PT_USE_ARCH_LR_DEFAULTS = _to_bool(neural['pt_use_arch_lr_defaults'])
    if 'pt_lr_unetplusplus' in neural:
        cfg.PT_LR_UNETPLUSPLUS = _to_float(neural['pt_lr_unetplusplus'])
    if 'pt_lr_deeplabv3plus' in neural:
        cfg.PT_LR_DEEPLABV3PLUS = _to_float(neural['pt_lr_deeplabv3plus'])
    if 'pt_lr_segformer' in neural:
        cfg.PT_LR_SEGFORMER = _to_float(neural['pt_lr_segformer'])
    if 'pt_weight_decay' in neural:
        cfg.PT_WEIGHT_DECAY = _to_float(neural['pt_weight_decay'])
    if 'pt_encoder_lr_mult' in neural:
        cfg.PT_ENCODER_LR_MULT = _to_float(neural['pt_encoder_lr_mult'])
    if 'pt_grad_clip_norm' in neural:
        cfg.PT_GRAD_CLIP_NORM = _to_float(neural['pt_grad_clip_norm'])
    if 'pt_loss' in neural:
        cfg.PT_LOSS = str(neural['pt_loss']).lower()
    if 'pt_loss_ce_weight' in neural:
        cfg.PT_LOSS_CE_WEIGHT = _to_float(neural['pt_loss_ce_weight'])
    if 'pt_loss_tversky_weight' in neural:
        cfg.PT_LOSS_TVERSKY_WEIGHT = _to_float(neural['pt_loss_tversky_weight'])
    if 'pt_tversky_alpha' in neural:
        cfg.PT_TVERSKY_ALPHA = _to_float(neural['pt_tversky_alpha'])
    if 'pt_tversky_beta' in neural:
        cfg.PT_TVERSKY_BETA = _to_float(neural['pt_tversky_beta'])
    if 'pt_tversky_gamma' in neural:
        cfg.PT_TVERSKY_GAMMA = _to_float(neural['pt_tversky_gamma'])
    if 'pt_boundary_loss_weight' in neural:
        cfg.PT_BOUNDARY_LOSS_WEIGHT = _to_float(neural['pt_boundary_loss_weight'])
    if 'pt_boundary_kernel_size' in neural:
        cfg.PT_BOUNDARY_KERNEL_SIZE = _to_int(neural['pt_boundary_kernel_size'])
    if 'pt_boundary_ignore_background' in neural:
        cfg.PT_BOUNDARY_IGNORE_BACKGROUND = _to_bool(neural['pt_boundary_ignore_background'])
    if 'pt_class_weights' in neural:
        cfg.PT_CLASS_WEIGHTS = _to_float_list(neural['pt_class_weights'])

    if 'pt_label_smoothing' in neural:
        cfg.PT_LABEL_SMOOTHING = _to_float(neural['pt_label_smoothing'])

    if 'pt_mixup_alpha' in neural:
        cfg.PT_MIXUP_ALPHA = _to_float(neural['pt_mixup_alpha'])
    if 'pt_train_sampler' in neural:
        cfg.PT_TRAIN_SAMPLER = str(neural['pt_train_sampler']).lower()
    if 'pt_sampler_target_pathology_fraction' in neural:
        cfg.PT_SAMPLER_TARGET_PATHOLOGY_FRACTION = _to_float(neural['pt_sampler_target_pathology_fraction'])
    if 'pt_sampler_pathology_min_pixels' in neural:
        cfg.PT_SAMPLER_PATHOLOGY_MIN_PIXELS = _to_int(neural['pt_sampler_pathology_min_pixels'])
    if 'pt_sampler_replacement' in neural:
        cfg.PT_SAMPLER_REPLACEMENT = _to_bool(neural['pt_sampler_replacement'])
    if 'pt_stain_aware_aug' in neural:
        cfg.PT_STAIN_AWARE_AUG = _to_bool(neural['pt_stain_aware_aug'])
    if 'pt_stain_hsv_prob' in neural:
        cfg.PT_STAIN_HSV_PROB = _to_float(neural['pt_stain_hsv_prob'])
    if 'pt_stain_hed_prob' in neural:
        cfg.PT_STAIN_HED_PROB = _to_float(neural['pt_stain_hed_prob'])
    if 'pt_stain_norm_prob' in neural:
        cfg.PT_STAIN_NORM_PROB = _to_float(neural['pt_stain_norm_prob'])
    if 'pt_stain_norm_blend_range' in neural:
        blend = _to_float_list(neural['pt_stain_norm_blend_range'])
        if len(blend) != 2:
            raise ValueError(
                f"pt_stain_norm_blend_range must have 2 values, got {neural['pt_stain_norm_blend_range']!r}"
            )
        cfg.PT_STAIN_NORM_BLEND_RANGE = (float(blend[0]), float(blend[1]))

    if 'pt_num_workers' in neural:
        cfg.PT_NUM_WORKERS = _to_int(neural['pt_num_workers'])
    if 'pt_save_every_epoch' in neural:
        cfg.PT_SAVE_EVERY_EPOCH = _to_bool(neural['pt_save_every_epoch'])
    if 'pt_checkpoint_metric' in neural:
        cfg.PT_CHECKPOINT_METRIC = str(neural['pt_checkpoint_metric']).lower()
    if 'pt_pathology_class_index' in neural:
        cfg.PT_PATHOLOGY_CLASS_INDEX = _to_int(neural['pt_pathology_class_index'])
    if 'pt_include_background_metrics' in neural:
        cfg.PT_INCLUDE_BACKGROUND_METRICS = _to_bool(neural['pt_include_background_metrics'])
    if 'pt_pathology_min_keep_ratio' in neural:
        cfg.PT_PATHOLOGY_MIN_KEEP_RATIO = _to_float(neural['pt_pathology_min_keep_ratio'])
    if 'pt_aug_skip_prob' in neural:
        cfg.PT_AUG_SKIP_PROB = _to_float(neural['pt_aug_skip_prob'])
    if 'pt_pathology_strict_aug' in neural:
        cfg.PT_PATHOLOGY_STRICT_AUG = _to_bool(neural['pt_pathology_strict_aug'])
    if 'pt_pathology_rotate_limit' in neural:
        cfg.PT_PATHOLOGY_ROTATE_LIMIT = _to_int(neural['pt_pathology_rotate_limit'])

    if 'pt_val_strategy' in neural:
        cfg.PT_VAL_STRATEGY = str(neural['pt_val_strategy']).lower()
    if 'pt_val_fraction' in neural:
        cfg.PT_VAL_FRACTION = _to_float(neural['pt_val_fraction'])
    if 'pt_val_seed' in neural:
        cfg.PT_VAL_SEED = _to_int(neural['pt_val_seed'])

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
    if 'tile_overlap' in dataset:
        cfg.TILE_OVERLAP = float(dataset['tile_overlap'])
    if 'centered_crop' in dataset:
        cfg.CENTERED_CROP = _to_bool(dataset['centered_crop'])
    if 'centered_crop_algo' in dataset:
        cfg.CENTERED_CROP_ALGO = str(dataset['centered_crop_algo']).lower()
    if 'centered_crop_edge_margin' in dataset:
        cfg.CENTERED_CROP_EDGE_MARGIN = _to_int(dataset['centered_crop_edge_margin'])
    if 'debug' in dataset:
        cfg.DATASET_DEBUG = _to_bool(dataset['debug'])
    if 'rect_border_padding' in dataset:
        cfg.RECT_BORDER_PADDING = _to_int(dataset['rect_border_padding'])
    if 'min_pathology_pixels' in dataset:
        cfg.MIN_PATHOLOGY_PIXELS = _to_int(dataset['min_pathology_pixels'])
    if 'force_extraction' in dataset:
        cfg.FORCE_EXTRACTION = _to_bool(dataset['force_extraction'])
    if 'normal_tiles_limit' in dataset:
        cfg.NORMAL_TILES_LIMIT_MODE = str(dataset['normal_tiles_limit']).lower()
    if 'normal_tiles_multiplier' in dataset:
        cfg.NORMAL_TILES_MULTIPLIER = _to_float(dataset['normal_tiles_multiplier'])

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


def _report_missing_yaml_keys(path: str, data: Mapping[str, Any], defaults: Config) -> None:
    """Print which supported YAML keys are missing and which defaults are used.

    This is intentionally informational: it does not change config behavior.
    """

    expected: dict[str, dict[str, str]] = {
        'general': {
            'current_slide': 'CURRENT_SLIDE',
            'current_slide_xml': 'CURRENT_SLIDE_XML',
            'openslide_path': 'OPENSLIDE_PATH',
            'hdd_slides': 'HDD_SLIDES',
            'hdd_slides_svs': 'HDD_SLIDES_SVS',
            'temp_folder': 'TEMP_FOLDER',
        },
        'neural_network': {
            'framework': 'FRAMEWORK',
            'dataset_folder': 'DATASET_FOLDER',
            'masks_folder': 'MASKS_FOLDER',
            'images_folder': 'IMAGES_FOLDER',
            'image_chunk': 'IMAGE_CHUNK',
            'image_shape': 'IMAGE_SHAPE',
            'classes': 'CLASSES',
            'pt_encoder_name': 'PT_ENCODER_NAME',
            'pt_model_arch': 'PT_MODEL_ARCH',
            'pt_encoder_weights': 'PT_ENCODER_WEIGHTS',
            'pt_use_encoder_preprocessing': 'PT_USE_ENCODER_PREPROCESSING',
            'pt_optimizer': 'PT_OPTIMIZER',
            'pt_scheduler': 'PT_SCHEDULER',
            'pt_scheduler_new': 'PT_SCHEDULER_NEW',
            'pt_scheduler_finetune': 'PT_SCHEDULER_FINETUNE',
            'pt_onecycle_pct_start': 'PT_ONECYCLE_PCT_START',
            'pt_lr': 'PT_LR',
            'pt_use_arch_lr_defaults': 'PT_USE_ARCH_LR_DEFAULTS',
            'pt_lr_unetplusplus': 'PT_LR_UNETPLUSPLUS',
            'pt_lr_deeplabv3plus': 'PT_LR_DEEPLABV3PLUS',
            'pt_lr_segformer': 'PT_LR_SEGFORMER',
            'pt_weight_decay': 'PT_WEIGHT_DECAY',
            'pt_encoder_lr_mult': 'PT_ENCODER_LR_MULT',
            'pt_grad_clip_norm': 'PT_GRAD_CLIP_NORM',
            'pt_loss': 'PT_LOSS',
            'pt_loss_ce_weight': 'PT_LOSS_CE_WEIGHT',
            'pt_loss_tversky_weight': 'PT_LOSS_TVERSKY_WEIGHT',
            'pt_tversky_alpha': 'PT_TVERSKY_ALPHA',
            'pt_tversky_beta': 'PT_TVERSKY_BETA',
            'pt_tversky_gamma': 'PT_TVERSKY_GAMMA',
            'pt_boundary_loss_weight': 'PT_BOUNDARY_LOSS_WEIGHT',
            'pt_boundary_kernel_size': 'PT_BOUNDARY_KERNEL_SIZE',
            'pt_boundary_ignore_background': 'PT_BOUNDARY_IGNORE_BACKGROUND',
            'pt_class_weights': 'PT_CLASS_WEIGHTS',
            'pt_label_smoothing': 'PT_LABEL_SMOOTHING',
            'pt_mixup_alpha': 'PT_MIXUP_ALPHA',
            'pt_train_sampler': 'PT_TRAIN_SAMPLER',
            'pt_sampler_target_pathology_fraction': 'PT_SAMPLER_TARGET_PATHOLOGY_FRACTION',
            'pt_sampler_pathology_min_pixels': 'PT_SAMPLER_PATHOLOGY_MIN_PIXELS',
            'pt_sampler_replacement': 'PT_SAMPLER_REPLACEMENT',
            'pt_stain_aware_aug': 'PT_STAIN_AWARE_AUG',
            'pt_stain_hsv_prob': 'PT_STAIN_HSV_PROB',
            'pt_stain_hed_prob': 'PT_STAIN_HED_PROB',
            'pt_stain_norm_prob': 'PT_STAIN_NORM_PROB',
            'pt_stain_norm_blend_range': 'PT_STAIN_NORM_BLEND_RANGE',
            'pt_num_workers': 'PT_NUM_WORKERS',
            'pt_save_every_epoch': 'PT_SAVE_EVERY_EPOCH',
            'pt_checkpoint_metric': 'PT_CHECKPOINT_METRIC',
            'pt_pathology_class_index': 'PT_PATHOLOGY_CLASS_INDEX',
            'pt_include_background_metrics': 'PT_INCLUDE_BACKGROUND_METRICS',
            'pt_pathology_min_keep_ratio': 'PT_PATHOLOGY_MIN_KEEP_RATIO',
            'pt_aug_skip_prob': 'PT_AUG_SKIP_PROB',
            'pt_pathology_strict_aug': 'PT_PATHOLOGY_STRICT_AUG',
            'pt_pathology_rotate_limit': 'PT_PATHOLOGY_ROTATE_LIMIT',
            'pt_val_strategy': 'PT_VAL_STRATEGY',
            'pt_val_fraction': 'PT_VAL_FRACTION',
            'pt_val_seed': 'PT_VAL_SEED',
            'labels': 'LABELS',
        },
        'gui': {
            'slide_dir': 'SLIDE_DIR',
            'unet_pred_mode': 'UNET_PRED_MODE',
            'use_tta': 'USE_TTA',
        },
        'dataset': {
            'exclude_duplicates': 'EXCLUDE_DUPLICATES',
            'broaden_individual_rect': 'BROADEN_INDIVIDUAL_RECT',
            'tile_overlap': 'TILE_OVERLAP',
            'centered_crop': 'CENTERED_CROP',
            'centered_crop_algo': 'CENTERED_CROP_ALGO',
            'centered_crop_edge_margin': 'CENTERED_CROP_EDGE_MARGIN',
            'debug': 'DATASET_DEBUG',
            'rect_border_padding': 'RECT_BORDER_PADDING',
            'min_pathology_pixels': 'MIN_PATHOLOGY_PIXELS',
            'force_extraction': 'FORCE_EXTRACTION',
            'normal_tiles_limit': 'NORMAL_TILES_LIMIT_MODE',
            'normal_tiles_multiplier': 'NORMAL_TILES_MULTIPLIER',
        },
        'web': {
            'ip_exposed': 'IP_EXPOSED',
            'endpoint_url': 'ENDPOINT_URL',
            'health_timeout': 'HEALTH_TIMEOUT',
            'web_conf_threshold': 'WEB_CONF_THRESHOLD',
            'atypical_class_index': 'WEB_ATYPICAL_CLASS_INDEX',
            'fast_tiles': 'WEB_FAST_TILES',
            'tile_cache_size': 'WEB_TILE_CACHE_SIZE',
            'scan_all_tissue': 'WEB_SCAN_ALL_TISSUE',
            'progress_every': 'WEB_PROGRESS_EVERY',
            'min_tissue_fraction': 'WEB_MIN_TISSUE_FRACTION',
            'tissue_gray_threshold': 'WEB_TISSUE_GRAY_THRESHOLD',
            'pt_batch_size': 'WEB_PT_BATCH_SIZE',
        },
    }

    missing_total = 0
    for section, keymap in expected.items():
        section_data = data.get(section, {})
        if not isinstance(section_data, Mapping):
            section_data = {}

        missing: list[tuple[str, Any]] = []
        for yaml_key, attr_name in keymap.items():
            if yaml_key not in section_data:
                missing.append((f'{section}.{yaml_key}', getattr(defaults, attr_name)))

        if missing:
            missing_total += len(missing)
            print(f"[config] Missing keys in '{path}' section '{section}'; using defaults:")
            for full_key, default_value in missing:
                print(f"  - {full_key}: {default_value!r}")

    if missing_total == 0:
        print(f"[config] All supported keys are present in '{path}'.")


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration into a Config object.

    This function has no side-effects besides an optional console print.
    """
    cfg = Config()
    defaults = Config()

    path = config_file or _find_default_config_file()
    if not path:
        return cfg

    if multiprocessing.current_process().name == 'MainProcess':
        print(f'{path} detected')

    if path.endswith(('.yaml', '.yml')):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        if multiprocessing.current_process().name == 'MainProcess':
            _report_missing_yaml_keys(path, data, defaults)
    elif path.endswith('.ini'):
        data = _load_ini_to_mapping(path)
    else:
        return cfg

    _apply_mapping(cfg, data)
    return cfg
