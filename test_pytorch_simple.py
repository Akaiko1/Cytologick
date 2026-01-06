#!/usr/bin/env python3
"""
Simple test script for PyTorch functionality validation.
Run independently to validate PyTorch AI pipeline.
"""

import sys
import os
sys.path.insert(0, '.')

from config import Config

def test_pytorch_dependencies():
    """Test PyTorch dependencies are available."""
    print("Testing PyTorch dependencies...")
    try:
        import torch
        import segmentation_models_pytorch as smp
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        print("✅ All PyTorch dependencies available")
        assert True
    except ImportError as e:
        print(f"❌ Missing PyTorch dependency: {e}")
        assert False, f"Missing PyTorch dependency: {e}"

def test_pytorch_model_creation():
    """Test PyTorch model creation."""
    print("Testing PyTorch model creation...")
    try:
        import segmentation_models_pytorch as smp
        
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        print("✅ U-Net model created successfully")
        assert model is not None
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        assert False, f"Model creation failed: {e}"

def test_pytorch_training_components():
    """Test PyTorch training components from our implementation."""
    print("Testing PyTorch training components...")
    try:
        # Import directly to avoid conflicts
        from clogic.ai_pytorch import (
            CombinedLoss,
            iou_score, f1_score, get_train_transforms
        )
        
        import torch
        
        cfg = Config()

        # Test transforms
        transform = get_train_transforms(cfg)
        print("✅ Data transforms work")
        
        # Test loss functions
        batch_size, num_classes, height, width = 2, 3, 32, 32
        predictions = torch.randn(batch_size, num_classes, height, width)
        targets = torch.randint(0, num_classes, (batch_size, height, width))
        
        combined_loss = CombinedLoss()
        c_loss = combined_loss(predictions, targets)

        print(f"✅ Loss function works: Combined={c_loss.item():.4f}")
        
        # Test metrics
        iou = iou_score(predictions, targets, num_classes)
        f1 = f1_score(predictions, targets, num_classes)
        print(f"✅ Metrics work: IoU={iou:.4f}, F1={f1:.4f}")
        
        assert True
    except Exception as e:
        print(f"❌ Training components failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Training components failed: {e}"

def test_pytorch_inference_components():
    """Test PyTorch inference components."""
    print("Testing PyTorch inference components...")
    try:
        from clogic.inference_pytorch import (
            image_to_tensor_pytorch, create_mask_pytorch, load_pytorch_model
        )
        import torch
        import numpy as np
        
        cfg = Config()

        # Test image preprocessing
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = image_to_tensor_pytorch(cfg, test_image)
        print(f"✅ Image preprocessing: {test_image.shape} -> {tensor.shape}")
        
        # Test mask creation
        pred = torch.randn(1, 3, 32, 32)
        mask = create_mask_pytorch(pred)
        print(f"✅ Mask creation: {pred.shape} -> {mask.shape}")
        
        assert True
    except Exception as e:
        print(f"❌ Inference components failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Inference components failed: {e}"

def test_end_to_end_pipeline():
    """Test end-to-end PyTorch pipeline."""
    print("Testing end-to-end PyTorch pipeline...")
    try:
        import torch
        import segmentation_models_pytorch as smp
        import numpy as np
        from clogic.inference_pytorch import apply_model_pytorch

        cfg = Config()
        
        # Create model
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        model.eval()
        
        # Test inference on sample image
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        
        with torch.no_grad():
            result = apply_model_pytorch(cfg, test_image, model, shapes=(128, 128))
        
        print(f"✅ End-to-end pipeline: {test_image.shape} -> {result.shape}")
        print(f"   Result range: [{result.min()}, {result.max()}]")
        
        assert True
    except Exception as e:
        print(f"❌ End-to-end pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"End-to-end pipeline failed: {e}"

def main():
    """Run all PyTorch tests."""
    print("🔬 Running Cytologick PyTorch AI Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_pytorch_dependencies,
        test_pytorch_model_creation,
        test_pytorch_training_components,
        test_pytorch_inference_components,
        test_end_to_end_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All PyTorch AI pipeline tests PASSED!")
        return 0
    else:
        print("⚠️  Some PyTorch tests FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())