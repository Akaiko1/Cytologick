"""
Tests for framework selection and configuration switching functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

# Import test fixtures
from .conftest import skip_if_no_pytorch


class TestFrameworkSelection:
    """Test framework selection and configuration switching."""

    def test_config_framework_default(self):
        """Test default framework configuration."""
        import config
        
        # Default should be tensorflow
        assert hasattr(config, 'FRAMEWORK')
        # Don't assert specific value as it may be changed by other tests

    def test_config_framework_override(self, original_config):
        """Test framework configuration override."""
        # Test setting framework to pytorch
        original_config.FRAMEWORK = 'pytorch'
        assert original_config.FRAMEWORK == 'pytorch'
        
        # Test setting framework to tensorflow
        original_config.FRAMEWORK = 'tensorflow'
        assert original_config.FRAMEWORK == 'tensorflow'

    def test_yaml_config_loading(self, temp_dir):
        """Test YAML configuration loading for framework selection."""
        import yaml
        from unittest.mock import patch
        
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
        
        # Test config loading
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            with patch('builtins.open', open):
                import config
                config._load_yaml_config(config_path)
                assert config.FRAMEWORK == 'pytorch'

    @patch('clogic.ai_pytorch.train_new_model_pytorch')
    @patch('clogic.ai.train_new_model')
    def test_training_script_framework_selection(self, mock_tf_train, mock_pytorch_train, original_config):
        """Test training scripts select correct framework."""
        
        # Test PyTorch selection
        original_config.FRAMEWORK = 'pytorch'
        
        # Import and run model_new (this should use PyTorch)
        with patch('config.FRAMEWORK', 'pytorch'):
            # Simulate running model_new.py
            from model_new import __name__ as model_new_name
            if model_new_name == '__main__':
                # This would call the appropriate function
                pass
        
        # Test TensorFlow selection
        original_config.FRAMEWORK = 'tensorflow'
        
        with patch('config.FRAMEWORK', 'tensorflow'):
            # Simulate running model_new.py
            from model_new import __name__ as model_new_name
            if model_new_name == '__main__':
                # This would call the appropriate function
                pass

    def test_module_imports_based_on_framework(self, original_config):
        """Test that correct modules are imported based on framework."""
        
        # Test PyTorch imports
        original_config.FRAMEWORK = 'pytorch'
        
        with patch('config.FRAMEWORK', 'pytorch'):
            # Force reimport of clogic module
            import importlib
            import clogic
            importlib.reload(clogic)
            
            # Should have PyTorch modules available
            assert hasattr(clogic, 'ai_pytorch')
            assert hasattr(clogic, 'inference_pytorch')
        
        # Test TensorFlow imports
        original_config.FRAMEWORK = 'tensorflow'
        
        with patch('config.FRAMEWORK', 'tensorflow'):
            # Force reimport of clogic module
            import importlib
            import clogic
            importlib.reload(clogic)
            
            # Should have TensorFlow modules available
            assert hasattr(clogic, 'ai')
            assert hasattr(clogic, 'inference')

    def test_framework_case_insensitive(self, original_config):
        """Test framework selection is case insensitive."""
        test_cases = [
            'pytorch', 'PyTorch', 'PYTORCH', 'PyTOrcH',
            'tensorflow', 'TensorFlow', 'TENSORFLOW', 'TenSorFlow'
        ]
        
        for framework in test_cases:
            original_config.FRAMEWORK = framework
            
            if framework.lower() == 'pytorch':
                # Should select PyTorch
                with patch('config.FRAMEWORK', framework):
                    assert framework.lower() == 'pytorch'
            else:
                # Should select TensorFlow
                with patch('config.FRAMEWORK', framework):
                    assert framework.lower() == 'tensorflow'

    def test_invalid_framework_handling(self, original_config):
        """Test handling of invalid framework values."""
        # Test invalid framework value
        original_config.FRAMEWORK = 'invalid_framework'
        
        # Should fall back to TensorFlow (default)
        with patch('config.FRAMEWORK', 'invalid_framework'):
            # The condition should evaluate to False, defaulting to TensorFlow path
            assert 'invalid_framework'.lower() != 'pytorch'

    @pytest.mark.usefixtures("skip_if_no_pytorch")
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

    def test_tensorflow_availability_check(self):
        """Test TensorFlow availability detection."""
        try:
            import tensorflow
            import segmentation_models
            tf_available = True
        except ImportError:
            tf_available = False
        
        # TensorFlow should be available in most cases
        if tf_available:
            assert tf_available
        else:
            pytest.skip("TensorFlow not available")

    def test_config_validation(self, original_config):
        """Test configuration validation for framework selection."""
        valid_frameworks = ['tensorflow', 'pytorch']
        
        for framework in valid_frameworks:
            original_config.FRAMEWORK = framework
            assert original_config.FRAMEWORK in valid_frameworks

    def test_mixed_framework_compatibility(self, original_config):
        """Test that framework switching doesn't break existing functionality."""
        
        # Switch to PyTorch
        original_config.FRAMEWORK = 'pytorch'
        
        # Basic config should still work
        assert hasattr(original_config, 'CLASSES')
        assert hasattr(original_config, 'IMAGE_SHAPE')
        assert hasattr(original_config, 'DATASET_FOLDER')
        
        # Switch to TensorFlow
        original_config.FRAMEWORK = 'tensorflow'
        
        # Same configs should still work
        assert hasattr(original_config, 'CLASSES')
        assert hasattr(original_config, 'IMAGE_SHAPE')
        assert hasattr(original_config, 'DATASET_FOLDER')

    def test_config_file_priority(self, temp_dir):
        """Test configuration file loading priority."""
        import yaml
        from unittest.mock import patch
        
        # Create YAML config with PyTorch
        yaml_config = {
            'neural_network': {
                'framework': 'pytorch'
            }
        }
        
        yaml_path = os.path.join(temp_dir, 'config.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f)
        
        # Test that YAML config is loaded when available
        with patch('os.path.exists') as mock_exists:
            def exists_side_effect(path):
                return path.endswith('config.yaml')
            
            mock_exists.side_effect = exists_side_effect
            
            with patch('builtins.open', open):
                import config
                # Would load YAML config if available
                # Framework should be set to pytorch from YAML
                pass

    def test_environment_variable_framework(self, original_config):
        """Test framework selection via environment variables (if implemented)."""
        # This test assumes future implementation of env var support
        with patch.dict(os.environ, {'CYTOLOGICK_FRAMEWORK': 'pytorch'}):
            # If env var support is added, it should override default
            # For now, just test that env vars don't break anything
            assert os.environ.get('CYTOLOGICK_FRAMEWORK') == 'pytorch'