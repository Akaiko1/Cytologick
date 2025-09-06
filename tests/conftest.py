"""
Pytest configuration and shared fixtures for Cytologick tests.
"""

import pytest
import os
import tempfile
import shutil
import numpy as np
from PIL import Image
import warnings
import torch

# Test data constants
TEST_IMAGE_SIZE = (128, 128, 3)
TEST_MASK_SIZE = (128, 128)
TEST_BATCH_SIZE = 2
TEST_NUM_CLASSES = 3


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 255, TEST_IMAGE_SIZE, dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample segmentation mask for testing."""
    return np.random.randint(0, TEST_NUM_CLASSES, TEST_MASK_SIZE, dtype=np.uint8)


@pytest.fixture
def sample_dataset_files(temp_dir, sample_image, sample_mask):
    """Create sample dataset files for testing."""
    images_dir = os.path.join(temp_dir, 'images')
    masks_dir = os.path.join(temp_dir, 'masks')
    os.makedirs(images_dir)
    os.makedirs(masks_dir)
    
    # Create sample files
    for i in range(5):
        # Save image
        img = Image.fromarray(sample_image)
        img.save(os.path.join(images_dir, f'image_{i}.bmp'))
        
        # Save mask
        mask = Image.fromarray(sample_mask)
        mask.save(os.path.join(masks_dir, f'image_{i}.bmp'))
    
    return {
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'num_files': 5
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    class MockConfig:
        FRAMEWORK = 'pytorch'
        DATASET_FOLDER = 'test_dataset'
        IMAGES_FOLDER = 'images'
        MASKS_FOLDER = 'masks'
        IMAGE_SHAPE = (128, 128)
        IMAGE_CHUNK = (256, 256)
        CLASSES = 3
        OPENSLIDE_PATH = '/mock/path'
        UNET_PRED_MODE = 'direct'
        
    return MockConfig()


@pytest.fixture
def pytorch_device():
    """Get available PyTorch device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Suppress noisy third-party DeprecationWarnings (protobuf upb types)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"Type google\._upb\._message\..*",
)


@pytest.fixture(scope="session")
def skip_if_no_pytorch():
    """Skip tests if PyTorch is not available."""
    try:
        import torch
        import segmentation_models_pytorch
        import albumentations
    except ImportError:
        pytest.skip("PyTorch dependencies not available")


@pytest.fixture
def original_config():
    """Backup and restore original config after test."""
    import config
    original_framework = getattr(config, 'FRAMEWORK', 'tensorflow')
    yield config
    config.FRAMEWORK = original_framework
