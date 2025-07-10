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
    
    pads = int(source.shape[0] % shapes[0]), int(source.shape[1] % shapes[1])
    source_pads = cv2.copyMakeBorder(
        source, 0, shapes[0] - pads[0], 0, shapes[1] - pads[1], cv2.BORDER_REPLICATE
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


def apply_model_raw_pytorch(source, model, classes, shapes=config.IMAGE_SHAPE):
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
    
    pads = int(source.shape[0] % shapes[0]), int(source.shape[1] % shapes[1])
    source_pads = cv2.copyMakeBorder(
        source, 0, shapes[0] - pads[0], 0, shapes[1] - pads[1], cv2.BORDER_REPLICATE
    )

    pathology_map = np.zeros((source_pads.shape[0], source_pads.shape[1], classes))

    with torch.no_grad():
        for x in range(0, source_pads.shape[0], shapes[0]):
            for y in range(0, source_pads.shape[1], shapes[1]):
                patch = source_pads[x: x + shapes[0], y: y + shapes[1]]
                
                # Convert patch to tensor and move to device
                patch_tensor = image_to_tensor_pytorch(patch).to(DEVICE)
                
                # Get prediction probabilities
                pred_mask = model(patch_tensor)
                pred_probs = F.softmax(pred_mask, dim=1)
                prediction = pred_probs[0].cpu().numpy().transpose(1, 2, 0)
                
                # Resize prediction back to patch size
                prediction_resized = cv2.resize(prediction, shapes, interpolation=cv2.INTER_LINEAR)
                
                pathology_map[x: x + shapes[0], y: y + shapes[1]] = prediction_resized

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
    model = smp.Unet(
        encoder_name='efficientnet-b3',
        encoder_weights='imagenet',
        classes=num_classes,
        activation='softmax2d'
    )
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    return model


def apply_model_smooth_pytorch(source, model, shape=config.IMAGE_SHAPE[0]):
    """
    Apply PyTorch model with smooth windowing to reduce edge artifacts.
    
    This function would require implementing the smooth windowing logic for PyTorch.
    For now, it falls back to the regular sliding window approach.
    
    Args:
        source: Input image as numpy array
        model: PyTorch model
        shape: Window size
        
    Returns:
        Pathology map as numpy array
    """
    # For now, use regular sliding window
    # TODO: Implement smooth windowing for PyTorch
    return apply_model_pytorch(source, model, (shape, shape))