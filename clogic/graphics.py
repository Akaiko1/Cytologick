import cv2

import numpy as np


def __get_contours(map, threshold):
    contours = cv2.findContours(map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    return [c for c in contours if cv2.contourArea(c) >= threshold]


def __get_probability(probability_map, contour) -> float:
    mask = np.zeros(probability_map.shape)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    return (float(np.sum(np.where(mask > 0, probability_map, 0))/np.sum(mask)) - 0.5) * 2


def __render_texts(texts: list, image) -> None:
    for t_x, t_y, text in texts:
        cv2.putText(image, text, 
            (t_x, t_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.55,
            (127, 0, 255, 255),
            2,
            -1)


def crop_dense_map(map, factor):
    height, width, _ = map.shape
    crop_h, crop_w = height%factor, width%factor

    return map[:height - crop_h,:width - crop_w]


def get_corrected_size(height, width, factor=256):
    crop_h, crop_w = height%factor, width%factor

    return height - crop_h, width - crop_w


def process_dense_pathology_map(pathology_map):
    markup = np.zeros((pathology_map[..., 0].shape + (4,)))
    red_marks = markup[..., 2].copy()
    stats, texts_probs, texts_labels = {}, [], []

    atypical_probability_map = pathology_map[..., 2]
    atypical_map = np.where(pathology_map[..., 2] >= 0.5, 1, 0)
    
    atypical_contours = __get_contours(atypical_map, threshold=500)

    for idx, cnt in enumerate(atypical_contours):
        stats[idx] = __get_probability(atypical_probability_map, cnt)
        cv2.drawContours(red_marks, [cnt], -1, 255, 2)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            texts_probs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), f'{stats[idx]:.2f} %'))
            texts_labels.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) - 20, f'label: {idx}'))


    markup[..., 1] = np.where(pathology_map[..., 1] >= 0.5, 255, 0)
    markup[..., 2] = red_marks
    
    markup[:, :, 3] = np.where(markup[:, :, 1] > 0, 64, markup[:, :, 3])  # green alpha channel transparency
    markup[:, :, 3] = np.where(markup[:, :, 2] > 0, 127, markup[:, :, 3])  # red alpha channel transparency

    for texts in [texts_labels, texts_probs]:
        __render_texts(texts, markup)

    return markup, stats


def process_sparse_pathology_map(pathology_map):
    markup = np.zeros((pathology_map.shape + (4,)))
    stats = {
        'Всего объектов':0,
        'Групп':0,
        'Предупреждений':0,
        'Атипичных':0,
    }

    all_cells = np.where(pathology_map != 0, 1, 0)
    atypical_parts = np.where(pathology_map == 2, 1, 0)

    cell_contours = cv2.findContours(all_cells.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    cell_contours = [c for c in cell_contours if cv2.contourArea(c) > 300]
    stats['Всего объектов'] = len(cell_contours)
    cv2.drawContours(markup, cell_contours, -1, (0, 255, 0), 2)

    single_cell_contours = [c for c in cell_contours if cv2.contourArea(c) <= 6500]
    cluster_contours = [c for c in cell_contours if cv2.contourArea(c) > 6500]
    stats['Групп'] = len(cluster_contours)

    atypical_contours = cv2.findContours(atypical_parts.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    atypical_contours = [a for a in atypical_contours if cv2.contourArea(a) > 300]
    
    # atypical_contours = contours.smooth_contours(atypical_contours, shape=markup.shape[:2])
    
    process_cells_sparse(markup, stats,
                        single_cell_contours, atypical_contours)
    process_clusters_sparse(markup, stats,
                            cluster_contours, atypical_contours)

    markup[:, :, 3] = np.where(markup[:, :, 2] > 0, 127, markup[:, :, 3])  # red alpha channel transparency
    markup[:, :, 3] = np.where(markup[:, :, 1] > 0, 200, markup[:, :, 3])  # green alpha channel transparency

    return markup, stats


def process_clusters_sparse(markup, stats, cluster_contours, atypical_contours):
    for cell in cluster_contours:
        detection = False
        for a_cell in atypical_contours:
            if cv2.pointPolygonTest(cell, (int(a_cell[0][0][0]), int(a_cell[0][0][1])), False) <= 0:
                continue
            
            detection = True
            cv2.drawContours(markup, [a_cell], -1, (0, 0, 255), -1)
        
            if detection:
                stats['Предупреждений'] += 1
                cv2.drawContours(markup, [cell], -1, (0, 255, 255), 2)


def process_cells_sparse(markup, stats, single_cell_contours, atypical_contours):
    for cell in single_cell_contours:
        proportion = 0
        for a_cell in atypical_contours:
            if cv2.pointPolygonTest(cell, (int(a_cell[0][0][0]), int(a_cell[0][0][1])), False) <= 0:
                continue

            proportion += int(cv2.contourArea(a_cell) * 100/cv2.contourArea(cell))

            if not proportion:
                continue

            stats['Атипичных'] += 1

            M = cv2.moments(cell)
            t_x, t_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            cv2.drawContours(markup, [cell], -1, (0, 0, 255), -1)
            cv2.drawContours(markup, [cell], -1, (0, 125, 255), 2)
            cv2.putText(markup, f'{proportion} %', 
                        (t_x, t_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.55,
                        (0, 1, 0),
                        2,
                        -1)
