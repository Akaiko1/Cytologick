"""
Tests for GUI integration with PyTorch models.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, Mock
import numpy as np

# Import test fixtures
from .conftest import skip_if_no_pytorch


@pytest.mark.usefixtures("skip_if_no_pytorch")
class TestGUIPyTorchIntegration:
    """Test GUI integration with PyTorch models."""

    def test_pytorch_model_loading_in_gui(self, temp_dir):
        """Test PyTorch model loading in GUI application."""
        from clogic.gui import Viewer
        import torch
        import segmentation_models_pytorch as smp
        
        # Create a test model and save it
        model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        
        # Create _main directory and save model
        main_dir = os.path.join(temp_dir, '_main')
        os.makedirs(main_dir)
        model_path = os.path.join(main_dir, 'model.pth')
        torch.save(model.state_dict(), model_path)
        
        # Mock config for PyTorch
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.CLASSES = 3
            mock_config.OPENSLIDE_PATH = '/mock/path'
            
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('clogic.gui.os.path.exists') as mock_gui_exists:
                    def exists_side_effect(path):
                        return path == model_path
                    mock_gui_exists.side_effect = exists_side_effect
                    
                    # Mock Qt components to avoid GUI creation
                    with patch('clogic.gui.QApplication'), \
                         patch('clogic.gui.Menu'), \
                         patch('clogic.gui.QWidget'):
                        
                        viewer = Viewer()
                        
                        # Check that model was loaded
                        assert viewer.model is not None

    def test_tensorflow_model_loading_in_gui(self, temp_dir):
        """Test TensorFlow model loading in GUI application."""
        from clogic.gui import Viewer
        
        # Create _main directory for TensorFlow model
        main_dir = os.path.join(temp_dir, '_main')
        os.makedirs(main_dir)
        
        # Mock config for TensorFlow
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'tensorflow'
            mock_config.OPENSLIDE_PATH = '/mock/path'
            
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('clogic.gui.tf.keras.models.load_model') as mock_load:
                    mock_model = MagicMock()
                    mock_load.return_value = mock_model
                    
                    # Mock Qt components
                    with patch('clogic.gui.QApplication'), \
                         patch('clogic.gui.Menu'), \
                         patch('clogic.gui.QWidget'):
                        
                        viewer = Viewer()
                        
                        # Check that TensorFlow model loading was called
                        mock_load.assert_called_once()
                        assert viewer.model == mock_model

    def test_gui_model_loading_failure_handling(self, temp_dir):
        """Test GUI handles model loading failures gracefully."""
        from clogic.gui import Viewer
        
        # Mock config for PyTorch
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.CLASSES = 3
            mock_config.OPENSLIDE_PATH = '/mock/path'
            
            # No model files exist
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = False
                
                # Mock Qt components
                with patch('clogic.gui.QApplication'), \
                     patch('clogic.gui.Menu'), \
                     patch('clogic.gui.QWidget'):
                    
                    viewer = Viewer()
                    
                    # Model should be None when loading fails
                    assert viewer.model is None

    def test_preview_window_pytorch_inference(self, temp_dir):
        """Test Preview window with PyTorch model inference."""
        from clogic.gui import Preview, Viewer
        from PyQt5.QtGui import QPixmap
        import torch
        import segmentation_models_pytorch as smp
        
        # Create mock parent viewer with PyTorch model
        parent = MagicMock(spec=Viewer)
        parent.model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        parent.model.eval()
        
        # Create test image file
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite('gui_preview.bmp', test_image)
        
        # Mock config for PyTorch
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.UNET_PRED_MODE = 'direct'
            
            # Mock PyTorch inference
            with patch('clogic.gui.inference.apply_model_pytorch') as mock_inference:
                mock_inference.return_value = np.zeros((128, 128), dtype=np.uint8)
                
                # Mock graphics processing
                with patch('clogic.gui.drawing.process_sparse_pathology_map') as mock_process:
                    mock_process.return_value = (np.zeros((128, 128, 3), dtype=np.uint8), {'test': 1})
                    
                    # Mock Qt components
                    with patch('clogic.gui.QWidget'), \
                         patch('clogic.gui.QPixmap'), \
                         patch('clogic.gui.cv2.imwrite'):
                        
                        # Create pixmap mock
                        pixmap = MagicMock(spec=QPixmap)
                        preview = Preview(parent, pixmap)
                        
                        # Test model inference
                        preview.runModel()
                        
                        # Verify PyTorch inference was called
                        mock_inference.assert_called_once()

    def test_preview_window_mode_selection(self, temp_dir):
        """Test Preview window mode selection with PyTorch."""
        from clogic.gui import Preview, Viewer
        from PyQt5.QtGui import QPixmap
        import torch
        import segmentation_models_pytorch as smp
        
        # Create mock parent viewer with PyTorch model
        parent = MagicMock(spec=Viewer)
        parent.model = smp.Unet(
            encoder_name='efficientnet-b3',
            encoder_weights='imagenet',
            classes=3,
            activation='softmax2d'
        )
        
        # Mock config
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.UNET_PRED_MODE = 'smooth'
            
            # Mock Qt components
            with patch('clogic.gui.QWidget'), \
                 patch('clogic.gui.QPixmap'), \
                 patch('clogic.gui.QRadioButton'):
                
                pixmap = MagicMock(spec=QPixmap)
                preview = Preview(parent, pixmap)
                
                # Test that modes are available
                assert hasattr(preview, 'modes')
                assert 'smooth' in preview.modes
                assert 'direct' in preview.modes
                assert 'remote' in preview.modes

    def test_gui_framework_switching(self, temp_dir):
        """Test GUI behavior when switching frameworks."""
        from clogic.gui import Viewer
        
        # Test switching from TensorFlow to PyTorch
        with patch('clogic.gui.config') as mock_config:
            mock_config.OPENSLIDE_PATH = '/mock/path'
            
            # Start with TensorFlow
            mock_config.FRAMEWORK = 'tensorflow'
            with patch('os.path.exists') as mock_exists, \
                 patch('clogic.gui.tf.keras.models.load_model') as mock_tf_load, \
                 patch('clogic.gui.QApplication'), \
                 patch('clogic.gui.Menu'), \
                 patch('clogic.gui.QWidget'):
                
                mock_exists.return_value = True
                mock_tf_load.return_value = MagicMock()
                
                viewer_tf = Viewer()
                assert viewer_tf.model is not None
            
            # Switch to PyTorch
            mock_config.FRAMEWORK = 'pytorch'
            with patch('os.path.exists') as mock_exists, \
                 patch('clogic.gui.inference.load_pytorch_model') as mock_pytorch_load, \
                 patch('clogic.gui.QApplication'), \
                 patch('clogic.gui.Menu'), \
                 patch('clogic.gui.QWidget'):
                
                mock_exists.return_value = True
                mock_pytorch_load.return_value = MagicMock()
                
                viewer_pytorch = Viewer()
                assert viewer_pytorch.model is not None

    def test_gui_remote_inference_compatibility(self, temp_dir):
        """Test that remote inference works regardless of local framework."""
        from clogic.gui import Preview, Viewer
        from PyQt5.QtGui import QPixmap
        
        # Create mock parent viewer
        parent = MagicMock(spec=Viewer)
        parent.model = None  # No local model
        
        # Create test image file
        test_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite('gui_preview.bmp', test_image)
        
        # Test remote inference with PyTorch framework
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.UNET_PRED_MODE = 'remote'
            
            # Mock remote inference (should use TensorFlow serving regardless)
            with patch('clogic.inference.apply_remote') as mock_remote:
                mock_remote.return_value = np.zeros((128, 128), dtype=np.uint8)
                
                with patch('clogic.gui.drawing.process_dense_pathology_map') as mock_process:
                    mock_process.return_value = (np.zeros((128, 128, 3), dtype=np.uint8), {'test': 1})
                    
                    with patch('clogic.gui.QWidget'), \
                         patch('clogic.gui.QPixmap'), \
                         patch('clogic.gui.cv2.imwrite'):
                        
                        pixmap = MagicMock(spec=QPixmap)
                        preview = Preview(parent, pixmap)
                        
                        # Test remote inference
                        preview.runModel()
                        
                        # Verify remote inference was called
                        mock_remote.assert_called_once()

    def test_gui_error_handling_with_pytorch(self, temp_dir):
        """Test GUI error handling with PyTorch models."""
        from clogic.gui import Viewer
        
        # Mock config for PyTorch
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.CLASSES = 3
            mock_config.OPENSLIDE_PATH = '/mock/path'
            
            # Mock model loading to raise exception
            with patch('clogic.gui.inference.load_pytorch_model') as mock_load:
                mock_load.side_effect = Exception("Model loading failed")
                
                with patch('os.path.exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    with patch('clogic.gui.QApplication'), \
                         patch('clogic.gui.Menu'), \
                         patch('clogic.gui.QWidget'):
                        
                        # Should not crash, model should be None
                        viewer = Viewer()
                        assert viewer.model is None

    @patch('builtins.print')  # Mock print to capture output
    def test_model_loading_messages(self, mock_print, temp_dir):
        """Test that appropriate messages are printed during model loading."""
        from clogic.gui import Viewer
        
        # Test PyTorch model loading message
        with patch('clogic.gui.config') as mock_config:
            mock_config.FRAMEWORK = 'pytorch'
            mock_config.CLASSES = 3
            mock_config.OPENSLIDE_PATH = '/mock/path'
            
            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True
                
                with patch('clogic.gui.inference.load_pytorch_model') as mock_load:
                    mock_load.return_value = MagicMock()
                    
                    with patch('clogic.gui.QApplication'), \
                         patch('clogic.gui.Menu'), \
                         patch('clogic.gui.QWidget'):
                        
                        viewer = Viewer()
                        
                        # Check that appropriate print messages were called
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        pytorch_messages = [msg for msg in print_calls if 'PyTorch' in msg]
                        assert len(pytorch_messages) > 0

    def test_preview_window_cleanup(self, temp_dir):
        """Test that Preview window cleans up properly."""
        # Clean up test files if they exist
        test_files = ['gui_preview.bmp', 'gui_map.png', 'gui_temp.bmp']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)