import cv2

import numpy as np


def __get_contours(map, threshold):
    contours = cv2.findContours(map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    return [c for c in contours if cv2.contourArea(c) >= threshold]


def __get_probability(probability_map, contour) -> float:
    mask = np.zeros(probability_map.shape)
    cv2.drawContours(mask, [contour], -1, 1, -1)
    # Average probability inside the contour (0..1)
    denom = max(1.0, float(np.sum(mask)))
    return float(np.sum(np.where(mask > 0, probability_map, 0)) / denom)


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


def process_dense_pathology_map(pathology_map, threshold: float = 0.5):
    markup = np.zeros((pathology_map[..., 0].shape + (4,)))
    red_marks = markup[..., 2].copy()
    stats, texts_probs, texts_labels = {}, [], []

    atypical_probability_map = pathology_map[..., 2]
    # Apply configurable threshold to detect lesions
    atypical_map = np.where(pathology_map[..., 2] >= float(threshold), 1, 0)
    
    atypical_contours = __get_contours(atypical_map, threshold=500)

    for idx, cnt in enumerate(atypical_contours):
        prob = __get_probability(atypical_probability_map, cnt)
        stats[idx] = prob
        cv2.drawContours(red_marks, [cnt], -1, 255, 2)

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            conf_pct = int(round(prob * 100))
            texts_probs.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), f'{conf_pct}%'))
            texts_labels.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]) - 20, f'lesion #{idx+1}'))


    # Keep background/other channel visualization at 0.5 threshold
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
        'Total objects': 0,
        'Groups': 0,
        'Warnings': 0,
        'Atypical': 0,
    }

    all_cells = np.where(pathology_map != 0, 1, 0)
    atypical_parts = np.where(pathology_map == 2, 1, 0)

    cell_contours = cv2.findContours(all_cells.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    cell_contours = [c for c in cell_contours if cv2.contourArea(c) > 300]
    stats['Total objects'] = len(cell_contours)
    cv2.drawContours(markup, cell_contours, -1, (0, 255, 0), 2)

    single_cell_contours = [c for c in cell_contours if cv2.contourArea(c) <= 6500]
    cluster_contours = [c for c in cell_contours if cv2.contourArea(c) > 6500]
    stats['Groups'] = len(cluster_contours)

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
                stats['Warnings'] += 1
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

            stats['Atypical'] += 1

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
