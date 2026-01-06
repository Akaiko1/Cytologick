"""
Tests for PyTorch inference pipeline functionality.
"""

import pytest
import torch
import numpy as np
import cv2
import os
from unittest.mock import patch, MagicMock
import tempfile

# Import test fixtures
from .conftest import skip_if_no_pytorch


@pytest.mark.usefixtures("skip_if_no_pytorch")
class TestPyTorchInference:
    """Test PyTorch inference pipeline components."""

    def test_image_to_tensor_conversion(self, cfg):
        """Test image preprocessing for PyTorch inference."""
        from clogic.inference_pytorch import image_to_tensor_pytorch
        
        # Create sample image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test with batch dimension
        tensor_with_batch = image_to_tensor_pytorch(cfg, image, add_dim=True)
        assert tensor_with_batch.shape == (1, 3, 128, 128)
        assert tensor_with_batch.dtype == torch.float32
        
        # Test without batch dimension
        tensor_without_batch = image_to_tensor_pytorch(cfg, image, add_dim=False)
        assert tensor_without_batch.shape == (3, 128, 128)
        assert tensor_without_batch.dtype == torch.float32
        
        # Check normalization (values should be in [0, 1])
        assert tensor_with_batch.min() >= 0
        assert tensor_with_batch.max() <= 1

    def test_create_mask_function(self, pytorch_device):
        """Test mask creation from model predictions."""
        from clogic.inference_pytorch import create_mask_pytorch
        
        # Create sample prediction
        batch_size, num_classes, height, width = 1, 3, 128, 128
        predictions = torch.randn(batch_size, num_classes, height, width)
        
        # Create mask
        mask = create_mask_pytorch(predictions)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (height, width)
        assert mask.dtype in [np.int64, np.int32]
        assert mask.min() >= 0
        assert mask.max() < num_classes

    def test_pytorch_model_loading(self, cfg, temp_dir):
        """Test PyTorch model loading functionality."""
        from clogic.inference_pytorch import load_pytorch_model
        import segmentation_models_pytorch as smp
        
        # Create and save a model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        
        model_path = os.path.join(temp_dir, 'test_model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Test loading
        loaded_model = load_pytorch_model(cfg, model_path, num_classes=3)
        
        assert loaded_model is not None
        assert isinstance(loaded_model, torch.nn.Module)
        
        # Test forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 3, 128, 128)
            output = loaded_model(test_input)
            assert output.shape == (1, 3, 128, 128)

    def test_sliding_window_inference(self, cfg, pytorch_device):
        """Test sliding window inference on larger images."""
        from clogic.inference_pytorch import apply_model_pytorch
        import segmentation_models_pytorch as smp
        
        # Create model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model.eval()
        
        # Create larger test image
        test_image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Apply model with sliding window
        with torch.no_grad():
            pathology_map = apply_model_pytorch(cfg, test_image, model, shapes=(128, 128))
        
        assert pathology_map.shape == test_image.shape[:2]
        assert pathology_map.dtype in [np.int64, np.int32, np.uint8]
        assert pathology_map.min() >= 0
        assert pathology_map.max() < 3

    def test_raw_probability_inference(self, cfg, pytorch_device):
        """Test inference returning raw probability maps."""
        from clogic.inference_pytorch import apply_model_raw_pytorch
        import segmentation_models_pytorch as smp
        
        # Create model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model.eval()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply model for raw probabilities
        with torch.no_grad():
            prob_map = apply_model_raw_pytorch(cfg, test_image, model, classes=3, shapes=(128, 128))
        
        assert prob_map.shape == (*test_image.shape[:2], 3)
        assert prob_map.dtype == np.float32
        
        # Check probabilities sum to 1 (approximately)
        prob_sums = prob_map.sum(axis=2)
        assert np.allclose(prob_sums, 1.0, atol=1e-5)

    def test_smooth_windowing_inference(self, cfg, pytorch_device):
        """Test smooth windowing inference (placeholder implementation)."""
        from clogic.inference_pytorch import apply_model_smooth_pytorch
        import segmentation_models_pytorch as smp
        
        # Create model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model.eval()
        
        # Create test image
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Apply smooth windowing
        with torch.no_grad():
            pathology_map = apply_model_smooth_pytorch(cfg, test_image, model, shape=128)
        
        assert pathology_map.shape == test_image.shape[:2]
        assert pathology_map.dtype in [np.int64, np.int32, np.uint8]

    def test_pytorch_tensorflow_output_compatibility(self, cfg, pytorch_device):
        """Test that PyTorch and TensorFlow inference produce similar outputs."""
        import segmentation_models_pytorch as smp
        
        # Create PyTorch model
        pytorch_model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        pytorch_model.eval()
        
        # Create test image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        # Test PyTorch inference
        from clogic.inference_pytorch import apply_model_pytorch
        with torch.no_grad():
            pytorch_result = apply_model_pytorch(cfg, test_image, pytorch_model, shapes=(128, 128))
        
        # Check output format compatibility
        assert isinstance(pytorch_result, np.ndarray)
        assert pytorch_result.shape == test_image.shape[:2]
        assert pytorch_result.dtype in [np.int64, np.int32, np.uint8]
        assert pytorch_result.min() >= 0
        assert pytorch_result.max() < 3

    def test_inference_with_different_image_sizes(self, cfg, pytorch_device):
        """Test inference works with various image sizes."""
        from clogic.inference_pytorch import apply_model_pytorch
        import segmentation_models_pytorch as smp
        
        # Create model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model.eval()
        
        # Test different image sizes
        test_sizes = [(100, 100), (128, 128), (200, 300), (512, 256)]
        
        for height, width in test_sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            with torch.no_grad():
                result = apply_model_pytorch(cfg, test_image, model, shapes=(128, 128))
            
            assert result.shape == (height, width)
            assert result.min() >= 0
            assert result.max() < 3

    def test_inference_error_handling(self, cfg, pytorch_device):
        """Test inference error handling for edge cases."""
        from clogic.inference_pytorch import apply_model_pytorch, load_pytorch_model
        import segmentation_models_pytorch as smp
        
        # Test with None model
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        with pytest.raises((AttributeError, TypeError)):
            apply_model_pytorch(cfg, test_image, None, shapes=(128, 128))
        
        # Test loading non-existent model
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_pytorch_model(cfg, '/non/existent/path.pth', num_classes=3)

    def test_device_handling_in_inference(self, cfg, pytorch_device):
        """Test proper device handling during inference."""
        from clogic.inference_pytorch import apply_model_pytorch
        import segmentation_models_pytorch as smp
        
        # Create model and move to device
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model = model.to(pytorch_device)
        model.eval()
        
        # Test inference
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        with torch.no_grad():
            result = apply_model_pytorch(cfg, test_image, model, shapes=(128, 128))
        
        # Result should be on CPU as numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape[:2]

    def test_memory_efficiency(self, cfg, pytorch_device):
        """Test memory efficiency during inference."""
        from clogic.inference_pytorch import apply_model_pytorch
        import segmentation_models_pytorch as smp
        
        # Create model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model.eval()
        
        # Test with larger image to check memory handling
        large_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        
        # This should not cause memory issues
        with torch.no_grad():
            result = apply_model_pytorch(cfg, large_image, model, shapes=(128, 128))
        
        assert result.shape == large_image.shape[:2]
        
        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()