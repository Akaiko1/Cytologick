import datetime
import itertools
import logging
from multiprocessing import Pool
from typing import List, Tuple, Optional, Callable

import numpy as np

from tfs_connector import tensorflow_layer as tfl
from tfs_connector.functions import split_evenly
from tfs_connector.image_manipulation import image_to_tensor
from tfs_connector.kmp_duplicate_lib_decorator import save_and_restore_kmp_duplicate_lib_ok
from tfs_connector.metrics import log_metrics
from tfs_connector.model_urls import get_model_predict_url


@log_metrics('tfs_connector.metrics')
def apply_classification_model_parallel(images: List[np.ndarray],
                                        model_name: str,
                                        model_version: int = 1,
                                        endpoint_url: str = 'http://localhost:8501',
                                        batch_size: int = 60,
                                        thread_count: int = 3,
                                        normalization: Callable[[np.ndarray], np.ndarray] = None,
                                        comfort_max_batch_size: Optional[int] = None) -> List[np.ndarray]:
    """
    Applies remote ANN classification model (served by TensorFlow Serving) to a list of images in a series of API
    requests, one request for each batch. The list of images is broken into multiple parts, each part processed in a
    separate thread. Images should be properly sized beforehand.

    :param images: List of images (np.ndarray) to process
    :param model_name: ANN model name on TFS
    :param model_version: ANN model version
    :param endpoint_url: TFS rest api endpoint URL
    :param batch_size: number of images smushed in a single query to ANN
    :param thread_count: number of parallel threads to run. If 0 or 1 passed single-threaded version of a function will
     be run.
    :param normalization: Normalization function to apply when image is converted into tensor
    :param comfort_max_batch_size: a comfort max batch size. If set Algorithm tries to divide images into
    thread_count batches, and if succeeds it will pass all images in one sitting, ignoring batch_size parameter
    :return: Results of ANN processing, list of probabilities of classes (np.ndarray)
    """

    main_logger = logging.getLogger('irym_tfs_connector.main')

    if thread_count == 0:
        main_logger.warning('Parameter thread_count passed == 0. Did you mean 1? '
                            'Running single-thread version without parallelism.')
        return apply_classification_model(images, model_name, model_version, endpoint_url, batch_size,
                                          normalization=normalization)

    if thread_count == 1:
        main_logger.warning('Parameter thread_count passed == 1. '
                            'Did you mean to use single-threaded version of a function? '
                            'Running single-thread version without parallelism.')
        return apply_classification_model(images, model_name, model_version, endpoint_url, batch_size,
                                          normalization=normalization)

    images_split = [x for x in split_evenly(images, thread_count)]

    if comfort_max_batch_size:
        max_len = max([len(x) for x in images_split])
        if max_len <= comfort_max_batch_size:
            main_logger.debug(f'comfort_max_batch_size set = {comfort_max_batch_size} '
                              f'and max count of images per thread = {max_len}. Setting batch size to {max_len} '
                              f'to perform ANN call in one sitting.')
            batch_size = max_len

    with Pool() as pool:
        params = [(idx, x, model_name, model_version, endpoint_url, batch_size, normalization) for idx, x
                  in enumerate(images_split)]
        result = pool.starmap(__apply_model_with_order, params)

    __sort_results(result)
    chained = itertools.chain.from_iterable((x[1] for x in result))

    return list(chained)


@save_and_restore_kmp_duplicate_lib_ok
@log_metrics('irym_tfs_connector.metrics')
def apply_classification_model(images: List[np.ndarray],
                               model_name: str,
                               model_version: int = 1,
                               endpoint_url: str = 'http://localhost:8501',
                               batch_size: int = 30,
                               normalization: Optional[Callable[[int, int], int]] = None) -> List[np.ndarray]:
    """
    Applies remote ANN classification model (served by TensorFlow Serving) to a list of images in a series of API
    requests, one request for each batch. Images should be properly sized beforehand.

    :param images: List of images (np.ndarray) to process
    :param model_name: ANN model name on TFS
    :param model_version: ANN model version
    :param endpoint_url: TFS rest api endpoint URL
    :param batch_size: number of images smushed in a single query to ANN
    :param normalization: Normalization function to apply when image is converted into tensor
    :return: Results of ANN processing, list of probabilities of classes (np.ndarray)
    """
    results = []

    pulse_logger = logging.getLogger('irym_tfs_connector.pulse')

    batches_bounds = __get_bounds_of_batches_by_size(images, batch_size)

    batches_count = len(batches_bounds)
    pulse_logger.debug(f'batch_size: {batch_size}, {len(batches_bounds)} batches')

    model_predict_url = get_model_predict_url(endpoint_url, model_name, model_version)

    for idx, bounds in enumerate(batches_bounds):
        pulse_logger.debug(f'Applying model to batch #{idx + 1} of {batches_count}. '
                           f'Length of a batch: {bounds[1] - bounds[0] + 1}.')
        started = datetime.datetime.utcnow()
        stack_results = __apply_classification_model_to_batch(images, bounds, model_predict_url, normalization)
        results.extend(stack_results)
        ended = datetime.datetime.utcnow()
        pulse_logger.debug(f'Batch #{idx + 1} of {batches_count} took {str(ended - started)}')

    return results


def __apply_model_with_order(thread_index: int, *args) -> Tuple[int, List[np.ndarray]]:
    """
    Proxy function that runs apply_classification_model on passed arguments to use in a separate thread. Is used to
    track batches of results from different threads and restore sorting afterwards

    :param thread_index: index of a thread
    :param args: arguments set that will be passed into apply_classification_model
    :return: List of resulting probabilities of classes (np.ndarray) grouped by thread index
    """
    return thread_index, apply_classification_model(*args)


def __get_bounds_of_batches_by_size(input_list: List, batch_size: int) -> List[Tuple[int, int]]:
    """
    Creates a list of batch bounds based on a list and size of a batch. Doesn't modify a list.
    :param input_list: A list that we need to split into batches
    :param batch_size: Size of a batch
    :return: List of bounds (lower bound, upper bound), zero-based, inclusive.

    """
    result = []
    for i in range(0, len(input_list), batch_size):
        lower_bound = i
        higher_bound = min(i + batch_size, len(input_list)) - 1
        result.append((lower_bound, higher_bound))
    return result


@log_metrics('irym_tfs_connector.metrics')
def __apply_classification_model_to_batch(all_images: List[np.ndarray], batch_bounds: Tuple[int, int],
                                          model_predict_url: str,
                                          normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) \
        -> List[np.ndarray]:
    """
    Applies remote TensorFlow Serving classification model to a batch of images

    :param all_images: Full list of images (to avoid extra copying)
    :param batch_bounds: Bounds of a batch, zero-based, inclusive. (0, 2) will pass items 0, 1 and 2.
    :param model_predict_url: TensorFlow Serving REST API Predict method of a chosen model
    :param normalization: Normalization function to apply when image is converted into tensor
    :return: List of resulting probabilities of classes
    """
    tensor_batch = []
    for idx in range(batch_bounds[0], batch_bounds[1] + 1):
        tensor_batch.append(image_to_tensor(all_images[idx], normalization))

    predictions = tfl.get_tensorflow_prediction(tensor_batch, model_predict_url)

    return [np.asarray(prediction, dtype=np.float32) for prediction in predictions]


@log_metrics('irym_tfs_connector.metrics')
def __sort_results(result: List[Tuple[int, List[np.ndarray]]]) -> None:
    """
    Sorts processing results by process number
    :param result: List of tuples (process number, list of resulting images)
    """
    result.sort(key=lambda x: x[0])


def parse_classification_results_into_classes_list(classification_results: List[np.ndarray]) -> np.ndarray:
    """
    Collapses classification results into a list of classes
    :param classification_results: TensorFlow classification prediction results
    :return: a list of classes
    """
    result = np.argmax(classification_results, -1).astype(np.uint16)
    return result
