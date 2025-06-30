import os
import time
import cv2
import config
import random
import clogic.graphics as graphics

import numpy as np
import tfs_connector as tfs
import matplotlib.pyplot as plt

from clogic import ai, inference

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(config.OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def get_slide(slide_path):
    return openslide.OpenSlide(slide_path)


def get_slide_rois(slide_path):
    slide = get_slide(slide_path)
    print(slide.level_downsamples, slide.level_dimensions, slide.level_count)

    region_level, region_dimensions, downsampling_coeff = slide.level_count - 2, slide.level_dimensions[slide.level_count - 2], slide.level_downsamples[slide.level_count - 2]
    preview = slide.read_region((0, 0), region_level, region_dimensions)
    preview = np.array(preview)

    preview_gray = cv2.cvtColor(preview, cv2.COLOR_RGBA2GRAY)
    preview_bin_gauss = cv2.adaptiveThreshold(preview_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(preview_bin_gauss, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # _ = plt.imshow(preview_bin_gauss)
    # plt.show()

    sample_size = (25, 25)
    probes_bin = []

    for x in range(0, region_dimensions[0], sample_size[0]):
        for y in range(0, region_dimensions[1], sample_size[1]):
                sample = preview_bin_gauss[y:y + sample_size[1], x:x + sample_size[0]]
                normalized_sum = np.sum(sample/255)

                if normalized_sum <= 225:
                     preview_bin_gauss[y:y + sample_size[1], x:x + sample_size[0]] = 0
                else:
                     probes_bin.append([x, y])
    
    # _ = plt.imshow(preview_bin_gauss)
    # plt.show()

    # x,y,w,h = cv2.boundingRect(sorted(contours, key=lambda x: cv2.contourArea(x))[-1])
    # cv2.rectangle(preview_bin_gauss, (x,y), (x+w,y+h), 255, 5)

    probes_bin_rescaled = [[int(e * downsampling_coeff) for e in coords] for coords in probes_bin]
    print(len(probes_bin_rescaled))

    selected_probes, selected_coordinate_shifts = [], []
    for probe in random.sample(probes_bin_rescaled, 1):

        probe_w, probe_h = int(sample_size[0] * downsampling_coeff), int(sample_size[1] * downsampling_coeff)
        probe_w, probe_h = graphics.get_corrected_size(probe_w, probe_h, 1024)

        print(f'Corrected size: {probe_w, probe_h}')

        for x in range(probe[0], probe[0] + probe_w, 1024):
             for y in range(probe[1], probe[1] + probe_h, 1024):
                probe_image = slide.read_region((x, y), 0, (1024, 1024))
                probe_image = np.array(probe_image)
                probe_image = cv2.cvtColor(probe_image, cv2.COLOR_RGBA2RGB)
                # _ = plt.imshow(probe_image)
                # plt.show()
                selected_coordinate_shifts.append((x, y))
                selected_probes.append(probe_image)
                # print(probe_image.shape)
    
    # only for calibration
    # selected_probes = [selected_probes[0]]
    # selected_coordinate_shifts = [selected_coordinate_shifts[0]]

    start = time.time()
    print(f'Selected probes quantity: {len(selected_probes)}')
    resize_ops = tfs.ResizeOptions(chunk_size=(256, 256), model_input_size=(128, 128))
    pathology_maps = tfs.apply_segmentation_model_parallel(selected_probes, endpoint_url='http://51.250.28.160:7500', model_name='demetra', batch_size=4,
                                                  resize_options=resize_ops, normalization=lambda x: x/255, parallelism_mode=1, thread_count=8)
    print(f'Remote execution took {time.time() - start} seconds')

    print(selected_coordinate_shifts)

    results = __get_cnts_from_regions(pathology_maps, selected_coordinate_shifts)

    # for idx, probe in enumerate(selected_probes):
    #     map = np.asarray(ai.create_mask([pathology_map[idx]]))[:probe.shape[0], :probe.shape[1]]
    #     ai.display([probe, map], tensors=False)

    return results


def __get_probability(probability_map, contour) -> float:
    mask = np.zeros(probability_map.shape)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    return (float(np.sum(np.where(mask > 0, probability_map, 0))/np.sum(mask)) - 0.5) * 2


def __get_cnts_from_regions(rmaps, shifts):
    result = {}

    for idx, rmap in enumerate(rmaps):

        atypical_probability_map = rmap[..., 2]
        atypical_map = np.where(rmap[..., 2] > 0.5, 255, 0)
        cnts = cv2.findContours(atypical_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        filtered_cnts = []
        for cnt in cnts:
            if __get_probability(atypical_probability_map, cnt) >= 0.95:
                filtered_cnts.append(cnt)

        if not filtered_cnts:
            continue

        shift = shifts[idx]
        for cnt in filtered_cnts:
            shifted = [r + shift for r in cnt]
            result[f'cnt_{idx}'] = [shifts[idx], [cv2.boundingRect(np.array(shifted))], [r.tolist() for r in shifted]]

    return result
