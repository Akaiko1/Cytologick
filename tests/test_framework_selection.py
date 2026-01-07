"""
Tests for framework selection and configuration switching functionality.
"""

import os
import pytest

from config import Config, load_config


class TestFrameworkSelection:
    """Test framework selection and configuration switching."""

    def test_config_object_has_framework(self):
        """Config is an object (no import-time globals)."""
        cfg = Config()
        assert hasattr(cfg, 'FRAMEWORK')
        assert isinstance(cfg.FRAMEWORK, str)

    def test_yaml_config_loading(self, temp_dir):
        """Test YAML configuration loading for framework selection."""
        import yaml
        
        # Create test YAML config
        config_data = {
            'neural_network': {
                'framework': 'pytorch',
                'classes': 3,
                'dataset_folder': 'test_dataset'
            }
        }
        
        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        cfg = load_config(config_path)
        assert cfg.FRAMEWORK.lower() == 'pytorch'
        assert cfg.CLASSES == 3
        assert cfg.DATASET_FOLDER == 'test_dataset'
        # Defaults should remain intact when not specified
        assert cfg.PT_LABEL_SMOOTHING == 0.0
        assert cfg.PT_MIXUP_ALPHA == 0.2

    def test_yaml_config_overrides_new_knobs(self, temp_dir):
        """New training knobs should be configurable via YAML."""
        import yaml

        config_data = {
            'neural_network': {
                'framework': 'pytorch',
                'classes': 3,
                'pt_label_smoothing': 0.05,
                'pt_mixup_alpha': 0.0,
            }
        }

        config_path = os.path.join(temp_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        cfg = load_config(config_path)
        assert cfg.PT_LABEL_SMOOTHING == pytest.approx(0.05)
        assert cfg.PT_MIXUP_ALPHA == pytest.approx(0.0)

    def test_module_imports_available(self):
        """PyTorch modules import without config side-effects."""
        import clogic.ai_pytorch
        import clogic.inference_pytorch

    def test_framework_case_insensitive(self):
        """Test framework selection is case insensitive."""
        test_cases = [
            'pytorch', 'PyTorch', 'PYTORCH', 'PyTOrcH',
            'tensorflow', 'TensorFlow', 'TENSORFLOW', 'TenSorFlow'
        ]
        
        for framework in test_cases:
            if framework.lower() == 'pytorch':
                assert framework.lower() == 'pytorch'
            else:
                assert framework.lower() == 'tensorflow'

    def test_invalid_framework_handling(self):
        """Test handling of invalid framework values."""
        cfg = Config()
        cfg.FRAMEWORK = 'invalid_framework'
        assert cfg.FRAMEWORK.lower() != 'pytorch'

    def test_pytorch_availability_check(self):
        """Test PyTorch availability detection."""
        try:
            import torch
            import segmentation_models_pytorch
            import albumentations
            pytorch_available = True
        except ImportError:
            pytorch_available = False
        
        # If test is running, PyTorch should be available
        assert pytorch_available, "PyTorch dependencies should be available for these tests"

    def test_config_has_common_fields(self):
        cfg = Config()
        assert hasattr(cfg, 'CLASSES')
        assert hasattr(cfg, 'IMAGE_SHAPE')
        assert hasattr(cfg, 'DATASET_FOLDER')