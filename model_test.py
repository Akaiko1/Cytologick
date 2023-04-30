import cv2

import torch
import torchvision

import numpy as np
import tensorflow as tf

from demetra import ai, inference
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def tf_test():
    source = cv2.imread('test.bmp', 1)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    model = tf.keras.models.load_model('demetra_main', compile=False)
    pathology_map = inference.apply_model(source, model, shapes=(256, 256))

    canvas = np.zeros(source.shape)
    canvas[..., 1] = np.where(pathology_map == 1, 255, 0)
    canvas[..., 2] = np.where(pathology_map == 2, 255, 0)
    cv2.imwrite('test_result.jpg', canvas)

    ai.display([source, pathology_map], tensors=False)


def torch_test():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())

    source = cv2.imread('mg.jpg', 1)
    result = source.copy()
    original_shape = source.shape
    source = cv2.resize(source, (500, 500))

    sam = sam_model_registry['vit_h'](checkpoint='vit_h.pth')
    sam.to(device='cuda')

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=5,  # Requires open-cv to run post-processing
    )

    masks = mask_generator.generate(source)
    masks = sorted(masks, key=lambda x: np.sum(x['segmentation']), reverse=True)
    print(len(masks))

    for roi in masks:
        mask = np.array(roi['segmentation'])
        mask = cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]))

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        color = np.random.choice(range(256), size=3).tolist()
        cv2.drawContours(result, contours, -1, color, 5)
    
    cv2.imwrite('test_map.png', result)



if __name__ == '__main__':
    tf_test()
    # torch_test()
