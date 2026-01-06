"""
PyTorch implementation of inference pipeline for Cytologick.

This module provides PyTorch equivalents to the TensorFlow inference functions,
maintaining identical functionality and output format.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import segmentation_models_pytorch as smp

import config


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_to_tensor_pytorch(image, add_dim=True):
    """
    Convert image to PyTorch tensor with proper preprocessing.
    
    Args:
        image: Input image as numpy array
        add_dim: Whether to add batch dimension
        
    Returns:
        Preprocessed tensor
    """
    # Normalize to [0, 1] and convert to tensor
    image = torch.from_numpy(image / 255.0).float()
    
    # Resize to model input shape
    image = F.interpolate(
        image.permute(2, 0, 1).unsqueeze(0), 
        size=config.IMAGE_SHAPE, 
        mode='bilinear', 
        align_corners=False
    )
    
    if not add_dim:
        image = image.squeeze(0)
    
    return image


def create_mask_pytorch(pred_mask):
    """
    Create a mask from PyTorch model predictions by taking the argmax.
    
    Args:
        pred_mask: Model prediction tensor
        
    Returns:
        Predicted mask as numpy array
    """
    pred_mask = torch.argmax(pred_mask, dim=1)
    return pred_mask[0].cpu().numpy()


def apply_model_pytorch(source, model, shapes=config.IMAGE_SHAPE):
    """
    Apply PyTorch model to image using sliding window approach.
    
    Args:
        source: Input image as numpy array
        model: PyTorch model
        shapes: Window size tuple
        
    Returns:
        Pathology map as numpy array
    """
    model.eval()
    model = model.to(DEVICE)
    
    # Compute minimal padding to fit whole tiles; pad=0 when divisible
    pad_h = (shapes[0] - (source.shape[0] % shapes[0])) % shapes[0]
    pad_w = (shapes[1] - (source.shape[1] % shapes[1])) % shapes[1]
    source_pads = cv2.copyMakeBorder(
        source, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
    )

    pathology_map = np.zeros(source_pads.shape[:2])

    with torch.no_grad():
        for x in range(0, source_pads.shape[0], shapes[0]):
            for y in range(0, source_pads.shape[1], shapes[1]):
                patch = source_pads[x: x + shapes[0], y: y + shapes[1]]
                
                # Convert patch to tensor and move to device
                patch_tensor = image_to_tensor_pytorch(patch).to(DEVICE)
                
                # Get prediction
                pred_mask = model(patch_tensor)
                prediction = create_mask_pytorch(pred_mask)
                
                # Resize prediction back to patch size
                prediction_resized = cv2.resize(
                    prediction.astype(np.uint8), shapes, interpolation=cv2.INTER_NEAREST
                )
                
                pathology_map[x: x + shapes[0], y: y + shapes[1]] = prediction_resized

    return pathology_map[:source.shape[0], :source.shape[1]].astype(np.uint8)


def apply_model_raw_pytorch(source, model, classes, shapes=config.IMAGE_SHAPE, batch_size: int | None = None):
    """
    Apply PyTorch model and return raw probability maps.
    
    Args:
        source: Input image as numpy array
        model: PyTorch model
        classes: Number of classes
        shapes: Window size tuple
        
    Returns:
        Raw probability map as numpy array
    """
    model.eval()
    model = model.to(DEVICE)
    
    pad_h = (shapes[0] - (source.shape[0] % shapes[0])) % shapes[0]
    pad_w = (shapes[1] - (source.shape[1] % shapes[1])) % shapes[1]
    source_pads = cv2.copyMakeBorder(
        source, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE
    )

    pathology_map = np.zeros((source_pads.shape[0], source_pads.shape[1], classes), dtype=np.float32)

    # Collect patches and their target locations
    coords: list[tuple[int,int]] = []
    tiles: list[np.ndarray] = []
    for x in range(0, source_pads.shape[0], shapes[0]):
        for y in range(0, source_pads.shape[1], shapes[1]):
            patch = source_pads[x: x + shapes[0], y: y + shapes[1]]
            tiles.append(patch)
            coords.append((x, y))

    # Determine batch size
    if batch_size is None:
        batch_size = 32 if torch.cuda.is_available() else 16

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_np = tiles[i:i+batch_size]
            # Build tensor batch [B, C, H, W]
            batch_t = torch.cat([image_to_tensor_pytorch(t) for t in batch_np], dim=0).to(DEVICE)
            logits = model(batch_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, C, H, W]
            probs = np.transpose(probs, (0, 2, 3, 1))       # [B, H, W, C]

            for j, pred in enumerate(probs):
                x, y = coords[i + j]
                # pred already at target size (shapes); place into map
                pathology_map[x: x + shapes[0], y: y + shapes[1]] = pred.astype(np.float32)

    return pathology_map[:source.shape[0], :source.shape[1]].astype(np.float32)


def load_pytorch_model(model_path: str, num_classes: int = 3):
    """
    Load a trained PyTorch model from file.
    
    Args:
        model_path: Path to the saved model state dict
        num_classes: Number of output classes
        
    Returns:
        Loaded PyTorch model
    """
    # Build model without pretrained weights; we will load user weights below
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights=None,
        classes=num_classes,
        activation=None  # return logits; softmax applied in inference
    )

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    
    return model


def apply_model_smooth_pytorch(source, model, shape=config.IMAGE_SHAPE[0]):
    """
    Apply PyTorch model with smooth windowing to reduce edge artifacts.
    
    Uses clogic.smooth to perform:
    1. Overlapping window prediction (reduces checkerboard artifacts)
    2. Test Time Augmentation (8x rotation/flip averaging)
    
    Args:
        source: Input image as numpy array
        model: PyTorch model
        shape: Window size
        
    Returns:
        Pathology map as numpy array
    """
    import clogic.smooth as smooth
    
    model.eval()
    model = model.to(DEVICE)

    # Define prediction wrapper for smooth.py
    # Input: numpy array (batch_size, H, W, C)
    # Output: numpy array (batch_size, H, W, n_classes)
    def _pred_func_pytorch(img_batch_subdiv):
        batch_size = 32 if torch.cuda.is_available() else 16
        outputs = []
        
        with torch.no_grad():
            for i in range(0, len(img_batch_subdiv), batch_size):
                batch_np = img_batch_subdiv[i:i+batch_size]
                
                # Convert to tensor: [B, H, W, C] -> [B, C, H, W]
                # Note: image_to_tensor_pytorch handles normalization /255.0
                batch_t = torch.cat([image_to_tensor_pytorch(t) for t in batch_np], dim=0).to(DEVICE)
                
                logits = model(batch_t)
                
                # Softmax and move to CPU
                probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, C, H, W]
                probs = np.transpose(probs, (0, 2, 3, 1))       # [B, H, W, C]
                outputs.append(probs)
                
        return np.concatenate(outputs, axis=0)

    # Run smooth prediction
    # use_tta=False gives ~8x speedup while still keeping smooth overlapping windows
    predictions_smooth = smooth.predict_img_with_smooth_windowing(
        source,
        window_size=shape,
        subdivisions=2,  # 50% overlap
        nb_classes=config.CLASSES,
        pred_func=_pred_func_pytorch,
        use_tta=config.USE_TTA
    )
    
    # Return probability map (H, W, C)
    return predictions_smooth.astype(np.float32)
