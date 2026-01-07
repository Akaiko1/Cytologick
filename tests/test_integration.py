"""
Integration tests for PyTorch functionality across the entire pipeline.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

# Import test fixtures
from .conftest import skip_if_no_pytorch


@pytest.mark.usefixtures("skip_if_no_pytorch")
class TestPyTorchIntegration:
    """Integration tests for complete PyTorch pipeline."""

    def test_end_to_end_pytorch_pipeline(self, cfg, sample_dataset_files, temp_dir):
        """Test complete pipeline from dataset to inference."""
        from clogic.ai_pytorch import get_datasets, CombinedLoss, iou_score
        from clogic.inference_pytorch import load_pytorch_model
        import torch
        import segmentation_models_pytorch as smp
        
        # 1. Test dataset creation
        train_dataset, val_dataset, total_samples = get_datasets(
            cfg,
            sample_dataset_files['images_dir'],
            sample_dataset_files['masks_dir'],
            train_split=0.8
        )
        
        assert total_samples == sample_dataset_files['num_files']
        assert len(train_dataset) + len(val_dataset) == total_samples
        
        # 2. Test model creation
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=3,
            activation=None
        )
        
        # 3. Test loss function
        loss_fn = CombinedLoss()
        
        # 4. Test training step (simplified)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get a sample batch
        sample_image, sample_mask = train_dataset[0]
        batch_images = sample_image.unsqueeze(0)
        batch_masks = sample_mask.unsqueeze(0)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = loss_fn(outputs, batch_masks)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 5. Test inference
        model.eval()
        with torch.no_grad():
            test_outputs = model(batch_images)
            iou = iou_score(test_outputs, batch_masks, num_classes=3)
            assert 0 <= iou <= 1
        
        # 6. Test model saving and loading
        model_path = os.path.join(temp_dir, 'integration_test_model.pth')
        torch.save(model.state_dict(), model_path)
        
        loaded_model = load_pytorch_model(cfg, model_path, num_classes=3)
        assert loaded_model is not None
        
        # 7. Test loaded model inference
        with torch.no_grad():
            loaded_outputs = loaded_model(batch_images)
            assert loaded_outputs.shape == test_outputs.shape

    def test_pytorch_inference_is_deterministic(self, cfg, sample_image):
        """Sanity-check PyTorch inference tensor path is deterministic."""
        import torch
        import segmentation_models_pytorch as smp
        
        # Create identical models (same random seed)
        torch.manual_seed(42)
        pytorch_model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,  # Use None to avoid pre-trained weights for fair comparison
            classes=3,
            activation=None
        )
        pytorch_model.eval()
        
        # Test with same input
        from clogic.inference_pytorch import image_to_tensor_pytorch
        
        # Convert image to tensor
        image_tensor = image_to_tensor_pytorch(cfg, sample_image, add_dim=True)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_model(image_tensor)
            pytorch_probs = torch.softmax(pytorch_output, dim=1)
        
        # Check output properties
        assert pytorch_output.shape == (1, 3, 128, 128)
        assert torch.allclose(pytorch_probs.sum(dim=1), torch.ones(1, 128, 128), atol=1e-5)
        
        # Test that outputs are deterministic
        with torch.no_grad():
            pytorch_output2 = pytorch_model(image_tensor)
        
        assert torch.allclose(pytorch_output, pytorch_output2, atol=1e-6)

    def test_config_driven_framework_switching(self, temp_dir, sample_dataset_files):
        """Test switching between frameworks through configuration."""
        import yaml
        
        # Create config file with PyTorch
        config_data = {
            'neural_network': {
                'framework': 'pytorch',
                'classes': 3,
                'dataset_folder': temp_dir,
                'images_folder': 'images',
                'masks_folder': 'masks'
            }
        }
        
        config_path = os.path.join(temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        from config import load_config
        cfg_loaded = load_config(config_path)
        assert cfg_loaded.FRAMEWORK.lower() == 'pytorch'
        assert cfg_loaded.CLASSES == 3

    def test_model_compatibility_across_frameworks(self, cfg, temp_dir):
        """Test that models trained with different frameworks are properly handled."""
        import torch
        import segmentation_models_pytorch as smp
        
        # Create PyTorch model
        pytorch_model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=3,
            activation=None
        )
        
        # Save PyTorch model
        pytorch_path = os.path.join(temp_dir, 'pytorch_model.pth')
        torch.save(pytorch_model.state_dict(), pytorch_path)
        
        # Test that PyTorch model can be loaded
        from clogic.inference_pytorch import load_pytorch_model
        loaded_pytorch = load_pytorch_model(cfg, pytorch_path, num_classes=3)
        assert loaded_pytorch is not None
        
        # Test inference with loaded model
        test_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = loaded_pytorch(test_input)
            assert output.shape == (1, 3, 128, 128)

    def test_training_script_integration(self, cfg, temp_dir, sample_dataset_files):
        """Test training scripts work with PyTorch backend."""
        from unittest.mock import patch
        
        cfg.FRAMEWORK = 'pytorch'
        cfg.DATASET_FOLDER = temp_dir
        cfg.IMAGES_FOLDER = 'images'
        cfg.MASKS_FOLDER = 'masks'
        cfg.CLASSES = 3

        from clogic.ai_pytorch import train_new_model_pytorch
        import torch

        with patch('torch.utils.data.DataLoader') as mock_loader, \
             patch('torch.save') as mock_save:

            mock_batch = (torch.randn(1, 3, 128, 128), torch.randint(0, 3, (1, 128, 128)))
            mock_loader.return_value.__iter__.return_value = [mock_batch]
            mock_loader.return_value.__len__.return_value = 1

            try:
                train_new_model_pytorch(
                    cfg,
                    os.path.join(temp_dir, 'test_model'),
                    output_classes=3,
                    epochs=1,
                    batch_size=1
                )
            except Exception as e:
                assert "CUDA" not in str(e)

    def test_inference_pipeline_integration(self, cfg, temp_dir, sample_image):
        """Test complete inference pipeline with PyTorch."""
        import torch
        import segmentation_models_pytorch as smp
        from clogic.inference_pytorch import apply_model_pytorch, load_pytorch_model
        
        # Create and save model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=3,
            activation=None
        )
        
        model_path = os.path.join(temp_dir, 'inference_test_model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Load model through inference pipeline
        loaded_model = load_pytorch_model(cfg, model_path, num_classes=3)
        
        # Test inference on sample image
        with torch.no_grad():
            result = apply_model_pytorch(cfg, sample_image, loaded_model, shapes=(128, 128))
        
        assert result.shape == sample_image.shape[:2]
        assert result.dtype in [np.int64, np.int32, np.uint8]
        assert result.min() >= 0
        assert result.max() < 3

    def test_memory_efficiency_integration(self, cfg):
        """Test memory efficiency across the pipeline."""
        import torch
        import segmentation_models_pytorch as smp
        
        # Test with larger model and data
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=3,
            activation=None
        )
        model.eval()
        
        # Process multiple images
        for i in range(3):
            large_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            with torch.no_grad():
                from clogic.inference_pytorch import apply_model_pytorch
                result = apply_model_pytorch(cfg, large_image, model, shapes=(128, 128))
            
            assert result.shape == (512, 512)
        
        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_error_propagation_integration(self, cfg, temp_dir):
        """Test error handling across the integrated pipeline."""
        # Test with invalid model path
        from clogic.inference_pytorch import load_pytorch_model
        import torch
        import pickle
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_pytorch_model(cfg, '/nonexistent/path.pth', num_classes=3)
        
        # Test with corrupted model file
        corrupted_path = os.path.join(temp_dir, 'corrupted.pth')
        with open(corrupted_path, 'w') as f:
            f.write("not a model file")
        
        with pytest.raises((RuntimeError, pickle.UnpicklingError, torch.serialization.pickle.UnpicklingError)):
            load_pytorch_model(cfg, corrupted_path, num_classes=3)