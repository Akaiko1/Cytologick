import cv2

import time
import logging
import numpy as np
import tensorflow as tf

import tfs_connector as tfs
from demetra import ai, inference

# import torch
# import torchvision
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def init_logging(logfile: str):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(logfile)])

    logging.getLogger('matplotlib.font_manager').propagate = False
    logging.getLogger('matplotlib.pyplot').propagate = False
    logging.getLogger('urllib3.connectionpool').propagate = False
    logging.getLogger('matplotlib').propagate = False


def tf_test():
    source = cv2.imread('test.bmp', 1)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    model = tf.keras.models.load_model('demetra_main', compile=False)
    pathology_map = inference.apply_model_raw(source, model, 3, shapes=(256, 256))

    canvas = np.zeros(source.shape)
    canvas[..., 1] = np.where(pathology_map[..., 1] >= 0.5, 255, 0)
    canvas[..., 2] = np.where(pathology_map[..., 2] >= 0.5, 255, 0)
    cv2.imwrite('test_result.jpg', canvas)

    ai.display([source, pathology_map], tensors=False)


def tf_test_remote():
    logger = logging.getLogger()
    init_logging('test.log')

    source = cv2.imread('test.bmp', 1)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

    start = time.time()
    resize_ops = tfs.ResizeOptions(chunk_size=(256, 256), model_input_size=(128, 128))
    pathology_map = tfs.apply_segmentation_model_parallel([source]*4, endpoint_url='http://194.113.237.3:7500', model_name='demetra', batch_size=2,
                                                  resize_options=resize_ops, normalization=lambda x: x/255, parallelism_mode=1, thread_count=4)
    pathology_map = np.asarray(ai.create_mask(pathology_map)[..., 0])[:source.shape[0], :source.shape[1]]
    print(f'Remote execution took {time.time() - start} seconds')

    # source = cv2.resize(source, (int(source.shape[1]/2), int(source.shape[0]/2)))
    # pathology_map = tfs.apply_smooth_tiling_segmentation_model(source, endpoint_url='http://194.113.237.3:7500', model_name='demetra',
    #                                               chunk_size=128, normalization=lambda x: x/255, parallelism_mode=1, thread_count=8)
    # pathology_map = np.asarray(ai.create_mask(pathology_map)[..., 0])

    canvas = np.zeros(source.shape)
    canvas[..., 1] = np.where(pathology_map == 1, 255, 0)
    canvas[..., 2] = np.where(pathology_map == 2, 255, 0)
    cv2.imwrite('test_result.jpg', canvas)

    ai.display([source, pathology_map], tensors=False)


# def torch_test():
#     print("PyTorch version:", torch.__version__)
#     print("Torchvision version:", torchvision.__version__)
#     print("CUDA is available:", torch.cuda.is_available())

#     source = cv2.imread('mg.jpg', 1)
#     result = source.copy()
#     original_shape = source.shape
#     source = cv2.resize(source, (500, 500))

#     sam = sam_model_registry['vit_h'](checkpoint='vit_h.pth')
#     sam.to(device='cuda')

#     mask_generator = SamAutomaticMaskGenerator(
#         model=sam,
#         points_per_side=32,
#         pred_iou_thresh=0.88,
#         stability_score_thresh=0.95,
#         crop_n_layers=0,
#         crop_n_points_downscale_factor=1,
#         min_mask_region_area=5,  # Requires open-cv to run post-processing
#     )

#     masks = mask_generator.generate(source)
#     masks = sorted(masks, key=lambda x: np.sum(x['segmentation']), reverse=True)
#     print(len(masks))

#     for roi in masks:
#         mask = np.array(roi['segmentation'])
#         mask = cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]))

#         contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
#         color = np.random.choice(range(256), size=3).tolist()
#         cv2.drawContours(result, contours, -1, color, 5)
    
#     cv2.imwrite('test_map.png', result)


if __name__ == '__main__':
    tf_test()
    # tf_test_remote()
    # torch_test()
