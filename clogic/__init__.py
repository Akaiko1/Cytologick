from .smooth import predict_img_with_smooth_windowing

# Framework-specific imports based on config
def _lazy_import():
    """Lazy import to avoid framework conflicts during testing."""
    try:
        import config
        if config.FRAMEWORK.lower() == 'pytorch':
            from . import ai_pytorch
            from . import inference_pytorch
        else:
            from . import ai
            from . import inference
    except ImportError:
        # Handle import issues gracefully
        pass

# Only do lazy import if not in testing environment
import os
if 'pytest' not in os.environ.get('_', '') and 'PYTEST_CURRENT_TEST' not in os.environ:
    _lazy_import()