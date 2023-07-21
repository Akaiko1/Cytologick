import cv2

import numpy as np


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
