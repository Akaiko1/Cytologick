from typing import List, Generator, Tuple

import numpy as np

from tfs_connector.metrics import log_metrics


@log_metrics('irym_tfs_connector.metrics')
def split_evenly(input_list: List, number_of_chunks: int) -> Generator[List, None, None]:
    """
    Splits a list into chunk of approximately equal size
    :param input_list: A list to split
    :param number_of_chunks: A number of chunks to split into
    :return: Generator of resulting lists
    """
    if number_of_chunks == 0 or number_of_chunks == 1:
        return (x for x in [input_list])
    quotinent, remainder = divmod(len(input_list), number_of_chunks)
    return (input_list[i * quotinent + min(i, remainder):(i + 1) * quotinent + min(i + 1, remainder)]
            for i in range(number_of_chunks))


@log_metrics('irym_tfs_connector.metrics')
def split_segmentation_results_by_classes(segmentation_results: np.ndarray) -> List[np.ndarray]:
    """
    Splits an ndarray [..., classes] into list of ndarrays, one ndarray for each probability class
    :param segmentation_results:
    :return:
    """
    return [segmentation_results[:, :, probabilities_class] for probabilities_class in
            range(segmentation_results.shape[-1])]


@log_metrics('irym_tfs_connector.metrics')
def parse_segmentation_results_into_probability_and_rois(segmentation_results: List[np.ndarray],
                                                         probabilities_class: int,
                                                         probability_treshold: float = 0.5) \
        -> Tuple[float, List[np.ndarray]]:
    """
    Parses TF ANN results into probability (total) and ROI (regions of interests)
    :param probability_treshold: Probability treshold for ROIs. Anything with probability less than this number will
     not be projected into ROIs.
    :param segmentation_results: A list of masks  you got from apply_segmentation_model or
    apply_segmentation_model_parallel
    :param probabilities_class: Index of a probabilities class to parse from segmentation results
    :return: (Total probability, List of ROI)
    """
    if probabilities_class < 0:
        raise AttributeError('Probabilities class param must be greater then 0')

    probabilities = [result[:, :, probabilities_class] for result in segmentation_results]
    probs = [np.where(r > probability_treshold, r, 0) for r in probabilities]
    rois = [np.where(p > 0, 1, 0) for p in probs]
    total_prob, total_area = sum([np.sum(p) for p in probs]), sum([np.sum(r) for r in rois])
    return (total_prob / total_area) if total_area else 0, rois
