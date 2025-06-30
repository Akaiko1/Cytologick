from typing import List, Callable

import numpy as np


def get_pretend_mirror_ann_function(nb_classes: int) -> Callable[[np.ndarray], List[np.ndarray]]:
    def __pretend_mirror_ann(_images: np.ndarray):
        return [x[..., :nb_classes] for x in _images]

    return __pretend_mirror_ann
