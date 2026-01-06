import os
import config
import logging
import warnings

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import clogic.inference_pytorch as inference_pytorch
except ImportError:
    inference_pytorch = None

def load_local_model():
    """
    Load a local model based on the configured framework.
    
    Returns:
        The loaded model object, or None if loading failed.
    """
    if config.FRAMEWORK.lower() == 'pytorch':
        return _load_pytorch_model()
    else:
        return _load_tensorflow_model()

def _load_pytorch_model():
    """Load PyTorch model from _main folder or fallback locations."""
    if inference_pytorch is None:
        print("PyTorch not installed or inference module missing")
        return None

    # Preferred model paths (in order of priority)
    model_files = [
        '_main/new_best.pth',      # Primary: best model from training (renamed)
        '_main/_new_best.pth',     # Auto-saved name from training
        '_main/new_final.pth',     
        '_main/_new_final.pth',
        '_main/new_last.pth',
        '_main/_new_last.pth',
        '_main/model.pth',         # Legacy paths
        '_main/model_best.pth', 
        '_main/model_final.pth'
    ]
    
    # Also fallback to any .pth file in _main if specific ones aren't found
    if os.path.isdir('_main'):
        for f in os.listdir('_main'):
            if f.endswith('.pth'):
                path = os.path.join('_main', f)
                if path not in model_files:
                    model_files.append(path)
    
    for model_path in model_files:
        if os.path.exists(model_path):
            print(f'Local PyTorch model located at {model_path}, loading')
            try:
                model = inference_pytorch.load_pytorch_model(model_path, config.CLASSES)
                print('PyTorch model loaded successfully')
                return model
            except Exception as e:
                print(f'Failed to load PyTorch model: {e}')
                continue
    
    print('No PyTorch model found in _main/ folder')
    return None

def _load_tensorflow_model():
    """Load TensorFlow model from _main folder."""
    if tf is None:
        print("TensorFlow not installed")
        return None

    warnings.warn(
        "TensorFlow local model loading is deprecated. Prefer PyTorch models and FRAMEWORK='pytorch'.",
        DeprecationWarning,
        stacklevel=2,
    )

    if os.path.exists('_main'):
        print('Local TensorFlow model located, loading')
        try:
            model = tf.keras.models.load_model('_main', compile=False)
            print('TensorFlow model loaded successfully')
            return model
        except Exception as e:
            print(f'Failed to load TensorFlow model: {e}')
            return None
    else:
        print('No TensorFlow model found in _main/ folder')
        return None
