"""
Tests for PyTorch training pipeline functionality.
"""

import pytest
import torch
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Import test fixtures
from .conftest import skip_if_no_pytorch


@pytest.mark.usefixtures("skip_if_no_pytorch")
class TestPyTorchTraining:
    """Test PyTorch training pipeline components."""

    def test_dataset_creation(self, cfg, sample_dataset_files):
        """Test PyTorch dataset creation from image/mask pairs."""
        from clogic.ai_pytorch import CytologyDataset, get_train_transforms
        
        dataset = CytologyDataset(
            sample_dataset_files['images_dir'],
            sample_dataset_files['masks_dir'],
            transform=get_train_transforms(cfg)
        )
        
        assert len(dataset) == sample_dataset_files['num_files']
        
        # Test dataset item retrieval
        image, mask = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape == (3, 128, 128)  # CHW format
        assert mask.shape == (128, 128)

    def test_data_transforms(self, cfg):
        """Test data augmentation transforms."""
        from clogic.ai_pytorch import get_train_transforms, get_val_transforms
        
        train_transform = get_train_transforms(cfg)
        val_transform = get_val_transforms(cfg)
        
        # Create sample data
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        mask = np.random.randint(0, 3, (256, 256), dtype=np.uint8)
        
        # Test training transforms
        train_result = train_transform(image=image, mask=mask)
        assert 'image' in train_result
        assert 'mask' in train_result
        assert train_result['image'].shape == (3, 128, 128)
        
        # Test validation transforms
        val_result = val_transform(image=image, mask=mask)
        assert val_result['image'].shape == (3, 128, 128)

    def test_loss_functions(self, pytorch_device):
        """Test custom loss functions."""
        from clogic.ai_pytorch import CombinedLoss
        
        batch_size, num_classes, height, width = 2, 3, 128, 128
        
        # Create sample predictions and targets
        predictions = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        # Test Combined Loss (Lovasz + CE)
        combined_loss = CombinedLoss()
        combined_value = combined_loss(predictions, targets)
        assert isinstance(combined_value, torch.Tensor)
        assert combined_value.item() >= 0
        
    def test_mixup_augmentation(self, pytorch_device):
        """Test Mixup augmentation."""
        from clogic.ai_pytorch import mixup_data
        
        batch_size = 4
        channels = 3
        height, width = 64, 64
        
        x = torch.randn(batch_size, channels, height, width).to(pytorch_device)
        y = torch.randint(0, 3, (batch_size, height, width)).to(pytorch_device)
        
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4, device=pytorch_device)
        
        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape
        assert isinstance(lam, (float, np.floating))
        assert 0 <= lam <= 1

    def test_metrics_calculation(self, pytorch_device):
        """Test IoU and F1 score calculation."""
        from clogic.ai_pytorch import iou_score, f1_score
        
        batch_size, num_classes = 2, 3
        
        # Deterministic targets with an absent class (class 2 is missing)
        targets = torch.tensor(
            [
                [[0, 1],
                 [0, 0]],
                [[1, 1],
                 [0, 0]],
            ],
            dtype=torch.long,
            device=pytorch_device,
        )

        # Case 1: perfect prediction => IoU=1, F1=1
        logits_perfect = torch.full((batch_size, num_classes, 2, 2), -10.0, device=pytorch_device)
        for b in range(batch_size):
            for i in range(2):
                for j in range(2):
                    cls = int(targets[b, i, j].item())
                    logits_perfect[b, cls, i, j] = 10.0

        iou = iou_score(logits_perfect, targets, num_classes)
        f1 = f1_score(logits_perfect, targets, num_classes)
        assert iou == pytest.approx(1.0, abs=1e-6)
        assert f1 == pytest.approx(1.0, abs=1e-6)

        # Case 2: predict all zeros for first sample (second sample perfect)
        logits_mixed = logits_perfect.clone()
        logits_mixed[0, :, :, :] = -10.0
        logits_mixed[0, 0, :, :] = 10.0

        iou2 = iou_score(logits_mixed, targets, num_classes)
        f12 = f1_score(logits_mixed, targets, num_classes)

        # Implementation aggregates over the whole batch (micro over batch), then averages over classes.
        # Across both samples:
        #   IoU0 = 5/6, IoU1 = 2/3, IoU2 = 1 (absent in both) => mean = 0.833333...
        #   F10  = 10/11, F11  = 0.8, F12  = 1 => mean = 0.903030...
        expected_iou = (5/6 + 2/3 + 1.0) / 3.0
        expected_f1 = (10/11 + 0.8 + 1.0) / 3.0
        assert iou2 == pytest.approx(expected_iou, abs=1e-5)
        assert f12 == pytest.approx(expected_f1, abs=1e-5)

    def test_model_architecture(self, pytorch_device):
        """Test U-Net model creation and forward pass."""
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=3,
            activation='softmax2d'
        )
        
        # Test model forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 128, 128)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 3, 128, 128)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size, 128, 128), atol=1e-5)

    @patch('clogic.ai_pytorch.get_datasets')
    @patch('torch.save')
    def test_train_new_model_pytorch(self, mock_save, mock_get_datasets, cfg, temp_dir):
        """Test training a new PyTorch model."""
        from clogic.ai_pytorch import train_new_model_pytorch
        
        # Mock dataset
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_get_datasets.return_value = (mock_train_dataset, mock_val_dataset, 100)
        
        # Mock data loader behavior
        sample_batch = (
            torch.randn(2, 3, 128, 128),
            torch.randint(0, 3, (2, 128, 128))
        )
        
        mock_train_dataset.__len__.return_value = 50
        mock_val_dataset.__len__.return_value = 25
        
        cfg.DATASET_FOLDER = temp_dir
        cfg.IMAGES_FOLDER = 'images'
        cfg.MASKS_FOLDER = 'masks'
        cfg.CLASSES = 3
        cfg.PT_ENCODER_WEIGHTS = None  # avoid downloading imagenet weights during tests

        # Test training with minimal epochs
        model_path = os.path.join(temp_dir, 'test_model')

        # This would normally train, but we'll mock the heavy parts
        # NOTE: ai_pytorch imports DataLoader into module namespace.
        with patch('clogic.ai_pytorch.DataLoader') as mock_loader:
            mock_loader.return_value.__iter__.return_value = [sample_batch]
            mock_loader.return_value.__len__.return_value = 1

            train_new_model_pytorch(cfg, model_path, 3, epochs=1, batch_size=2)
            assert mock_save.called

    def test_dataset_splitting(self, cfg, sample_dataset_files):
        """Test train/validation dataset splitting."""
        from clogic.ai_pytorch import get_datasets
        
        train_dataset, val_dataset, total_samples = get_datasets(
            cfg,
            sample_dataset_files['images_dir'],
            sample_dataset_files['masks_dir'],
            train_split=0.8
        )
        
        assert total_samples == sample_dataset_files['num_files']
        # With 5 files and 0.8 split: train=4, val=1
        assert len(train_dataset) == 4
        assert len(val_dataset) == 1

    def test_model_device_placement(self, pytorch_device):
        """Test model can be moved to available device."""
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            classes=3,
            activation='softmax2d'
        )
        
        # Test device placement
        model = model.to(pytorch_device)
        
        # Verify model is on correct device
        for param in model.parameters():
            assert param.device.type == pytorch_device.type