import numpy as np
import torch

from config import Config
from clogic.ai_pytorch import get_val_transforms
from clogic.inference_pytorch import image_to_tensor_pytorch


def test_train_infer_preprocessing_match_default() -> None:
    """Ensure train(val) and inference preprocessing are identical.

    This test targets the default project behavior (scale to [0, 1]).
    If encoder-specific preprocessing is enabled via config, both
    pipelines still share the same function and should remain consistent.
    """
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(301, 407, 3), dtype=np.uint8)

    cfg = Config()

    # Albumentations val pipeline
    transformed = get_val_transforms(cfg)(image=image, mask=np.zeros(image.shape[:2], dtype=np.uint8))
    train_t = transformed["image"]  # CHW float tensor

    # Inference preprocessing
    infer_t = image_to_tensor_pytorch(cfg, image, add_dim=False)  # CHW float tensor

    torch.testing.assert_close(train_t, infer_t, rtol=0.0, atol=1e-5)
