import datetime
import logging
import platform
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Callable

import multiprocess.pool
import numpy as np

warnings.warn(
    "tfs_connector.classification (TensorFlow Serving) is deprecated.",
    DeprecationWarning,
    stacklevel=2,
)

from tfs_connector import tensorflow_layer as tfl
from tfs_connector.functions import split_evenly
from tfs_connector.image_manipulation import image_to_tensor
from tfs_connector.kmp_duplicate_lib_decorator import save_and_restore_kmp_duplicate_lib_ok
from tfs_connector.metrics import log_metrics
from tfs_connector.model_urls import get_model_predict_url
from tfs_connector.parallelism_mode import ParallelismMode


@log_metrics('irym_tfs_connector.metrics')
def apply_classification_model_parallel(images: List[np.ndarray],
                                        model_name: str,
                                        model_version: int = 1,
                                        endpoint_url: str = 'http://localhost:8501',
                                        batch_size: int = 60,
                                        thread_count: int = 3,
                                        parallelism_mode: int = 0,
                                        normalization: Callable[[np.ndarray], np.ndarray] = None,
                                        comfort_max_batch_size: Optional[int] = None) -> List[np.ndarray]:
    """
    Applies remote ANN classification model (served by TensorFlow Serving) to a list of images in a series of API
    requests, one request for each batch. The list of images is broken into multiple parts, each part processed in a
    separate thread. Images should be properly sized beforehand.

    :param parallelism_mode: parallelism mode. Use ParallelismMode.PARALLELISM_MODE_* constants to fill correctly.
    Default multiprocess.
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

    if parallelism_mode not in ParallelismMode.ALLOWED_PARALLELISM_MODES:
        raise ValueError(f'Parameter parallelism_mode={parallelism_mode} has value that is not '
                         f'in allowed parallelism modes={ParallelismMode.ALLOWED_PARALLELISM_MODES}')

    if platform.system() == "Windows" and parallelism_mode == ParallelismMode.PARALLELISM_MODE_MULTIPROCESS:
        main_logger.warning('Multiprocess parallelism mode not well-suited for Windows system. '
                            'Try use ParallelismMode.PARALLELISM_MODE_THREADPOOL.')

    elif platform.system() == "Linux" and parallelism_mode == ParallelismMode.PARALLELISM_MODE_THREADPOOL:
        main_logger.warning('Multiprocess parallelism mode not well-suited for Windows system. '
                            'Try use ParallelismMode.PARALLELISM_MODE_MULTIPROCESS.')

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

    result = []

    if parallelism_mode == ParallelismMode.PARALLELISM_MODE_MULTIPROCESS:
        with multiprocess.pool.Pool(len(images_split)) as pool:
            mpp_results = [pool.apply_async(apply_classification_model, (images_slice, model_name, model_version, endpoint_url, batch_size, normalization)) for images_slice in images_split]

            for mpp_result in mpp_results:
                result.extend(mpp_result.get())

    elif parallelism_mode == ParallelismMode.PARALLELISM_MODE_THREADPOOL:
        with ThreadPoolExecutor(max_workers=len(images_split)) as executor:
            futures = [executor.submit(apply_classification_model, images_slice, model_name, model_version, endpoint_url,
                                       batch_size, normalization)
                       for images_slice in images_split]

            for future in futures:
                result.extend(future.result())

    return result


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


def parse_classification_results_into_classes_list(classification_results: List[np.ndarray]) -> np.ndarray:
    """
    Collapses classification results into a list of classes
    :param classification_results: TensorFlow classification prediction results
    :return: a list of classes
    """
    result = np.argmax(classification_results, -1).astype(np.uint16)
    return result
