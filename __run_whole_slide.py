import os
import time
import cv2
import config
import random

import numpy as np
import tfs_connector as tfs
import matplotlib.pyplot as plt

from demetra import ai, inference

with os.add_dll_directory(config.OPENSLIDE_PATH):
    import openslide


def get_slide(slide_path):
    return openslide.OpenSlide(slide_path)


def main():
    slide = get_slide(config.CURRENT_SLIDE)

    print(slide.level_downsamples, slide.level_dimensions, slide.level_count)

    region_level, region_dimensions, downsampling_coeff = slide.level_count - 2, slide.level_dimensions[slide.level_count - 2], slide.level_downsamples[slide.level_count - 2]
    preview = slide.read_region((0, 0), region_level, region_dimensions)
    preview = np.array(preview)

    preview_gray = cv2.cvtColor(preview, cv2.COLOR_RGBA2GRAY)
    preview_bin_gauss = cv2.adaptiveThreshold(preview_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(preview_bin_gauss, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sample_size = (50, 50)
    probes_bin = []

    for x in range(0, region_dimensions[0], sample_size[0]):
        for y in range(0, region_dimensions[1], sample_size[1]):
                sample = preview_bin_gauss[y:y + sample_size[1], x:x + sample_size[0]]
                normalized_sum = np.sum(sample/255)

                if normalized_sum <= 700:
                     preview_bin_gauss[y:y + sample_size[1], x:x + sample_size[0]] = 0
                else:
                     probes_bin.append([x, y])

    # x,y,w,h = cv2.boundingRect(sorted(contours, key=lambda x: cv2.contourArea(x))[-1])
    # cv2.rectangle(preview_bin_gauss, (x,y), (x+w,y+h), 255, 5)

    probes_bin_rescaled = [[int(e * downsampling_coeff) for e in coords] for coords in probes_bin]
    print(len(probes_bin_rescaled))

    for probe in [random.choice(probes_bin_rescaled)]:
        probe_image = slide.read_region((probe[0], probe[1]), 0, (int(sample_size[0] * downsampling_coeff), int(sample_size[1] * downsampling_coeff)))
        probe_image = np.array(probe_image)
        probe_image = cv2.cvtColor(probe_image, cv2.COLOR_RGBA2RGB)
        _ = plt.imshow(probe_image)
        plt.show()
        print(probe_image.shape)
    
    start = time.time()
    resize_ops = tfs.ResizeOptions(chunk_size=(256, 256), model_input_size=(128, 128))
    pathology_map = tfs.apply_segmentation_model_parallel([probe_image] * 4, endpoint_url='http://194.113.237.3:7500', model_name='demetra', batch_size=1,
                                                  resize_options=resize_ops, normalization=lambda x: x/255, parallelism_mode=1, thread_count=4)
    pathology_map = np.asarray(ai.create_mask(pathology_map)[..., 0])[:probe_image.shape[0], :probe_image.shape[1]]
    print(f'Remote execution took {time.time() - start} seconds')
    ai.display([probe_image, pathology_map], tensors=False)

    _ = plt.imshow(preview_bin_gauss)
    plt.show()


if __name__ == '__main__':
    main()
