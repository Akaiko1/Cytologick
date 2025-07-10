from .smooth import predict_img_with_smooth_windowing

# Framework-specific imports based on config
import config
if config.FRAMEWORK.lower() == 'pytorch':
    from . import ai_pytorch
    from . import inference_pytorch
else:
    from . import ai
    from . import inference