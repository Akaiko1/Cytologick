import datetime
import itertools
import logging
from multiprocessing import Pool
from typing import List, Tuple, Optional, Callable

import numpy as np

from tfs_connector import tensorflow_layer as tfl
from tfs_connector.functions import split_evenly
from tfs_connector.image_manipulation import resize_image, \
    cut_image_into_chunks, pad_images
from tfs_connector.kmp_duplicate_lib_decorator import save_and_restore_kmp_duplicate_lib_ok
from tfs_connector.metadata import get_model_input_size_from_metadata
from tfs_connector.metrics import log_metrics
from tfs_connector.model_urls import get_model_predict_url


@log_metrics('tfs_connector.metrics')
def parse_segmentation_results_into_probability_and_rois(results: List[np.ndarray], probabilities_class: int) \
        -> Tuple[float, List[np.ndarray]]:
    """
    Parses TF ANN results into probability (total) and ROI (regions of interests)
    :param results: A list of masks  you got from apply_segmentation_model or apply_segmentation_model_parallel
    :return: (Total probability, List of ROI)
    """
    if probabilities_class < 0:
        raise AttributeError('Probabilities class param must be greater then 0')

    probabilities = [result[:, :, probabilities_class] for result in results]
    probs = [np.where(r > 0.5, r, 0) for r in probabilities]
    rois = [np.where(p > 0, 1, 0) for p in probs]
    total_prob, total_area = sum([np.sum(p) for p in probs]), sum([np.sum(r) for r in rois])
    return (total_prob / total_area) if total_area else 0, rois


@log_metrics('irym_tfs_connector.metrics')
def apply_segmentation_model_parallel(images: List[np.ndarray],
                                      model_name: str,
                                      endpoint_url: str,
                                      model_version: int = 1,
                                      batch_size: int = 60,
                                      chunk_size: Optional[Tuple[int, int]] = None,
                                      model_input_size: Optional[Tuple[int, int]] = None,
                                      normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                                      thread_count: int = 3,
                                      comfort_max_batch_size: Optional[int] = None) -> List[np.ndarray]:
    """
    Applies remote ANN segmentation model (served by TensorFlow Serving) to a list of images in a series of API
    requests, one request for each batch. The list of images is broken into multiple parts, each part processed in a
    separate thread. In each thread each image is cut into chunks according to chunk_size parameter, then each chunk
    will be resized to model_input_size. All chunks of a batch will be smushed together and sent to TFS to process.

    :param images: List of images (np.ndarray) to process
    :param model_name: ANN model name on TFS
    :param model_version: ANN model version
    :param endpoint_url: TFS rest api endpoint URL
    :param batch_size: number of images smushed in a single query to ANN
    :param chunk_size: size of a chunk each image would be cut into. If passed None will be set to default (256, 256).
    :param model_input_size: size of a model input (2D). If None it will be computed from ANN metadata
    :param thread_count: number of parallel threads to run. If 0 or 1 passed single-threaded version of a function will
     be run.
    :param normalization: Normalization function to apply when image is converted into tensor
    :param comfort_max_batch_size: a comfort max batch size. If set Algorithm tries to divide images into
    thread_count batches, and if succeeds it will pass all images in one sitting, ignoring batch_size parameter
    :return: Results of ANN processing, list of images (np.ndarray)
    """

    main_logger = logging.getLogger('irym_tfs_connector.main')

    if thread_count == 0:
        main_logger.warning('Parameter thread_count passed == 0. Did you mean 1? '
                            'Running single-thread version without parallelism.')
        return apply_segmentation_model(images, endpoint_url, model_name, model_version, batch_size, chunk_size,
                                        model_input_size, normalization=normalization)

    if thread_count == 1:
        main_logger.warning('Parameter thread_count passed == 1. '
                            'Did you mean to use single-threaded version of a function? '
                            'Running single-thread version without parallelism.')
        return apply_segmentation_model(images, endpoint_url, model_name, model_version, batch_size, chunk_size,
                                        model_input_size, normalization=normalization)

    images_split = [x for x in split_evenly(images, thread_count)]

    if comfort_max_batch_size:
        max_len = max([len(x) for x in images_split])
        if max_len <= comfort_max_batch_size:
            main_logger.debug(f'comfort_max_batch_size set = {comfort_max_batch_size} '
                              f'and max count of images per thread = {max_len}. Setting batch size to {max_len} '
                              f'to perform ANN call in one sitting.')
            batch_size = max_len

    with Pool() as pool:
        result = pool.starmap(__apply_model_with_order, [
            (idx, images_slice, endpoint_url, model_name, model_version,
             batch_size, chunk_size, model_input_size, normalization) for idx, images_slice in
            enumerate(images_split)])

    __sort_results(result)
    chained = itertools.chain.from_iterable((x[1] for x in result))

    return list(chained)


@save_and_restore_kmp_duplicate_lib_ok
@log_metrics('irym_tfs_connector.metrics')
def apply_segmentation_model(images: List[np.ndarray],
                             endpoint_url: str,
                             model_name: str,
                             model_version: int = 1,
                             batch_size: int = 30,
                             chunk_size: Optional[Tuple[int, int]] = None,
                             model_input_size: Optional[Tuple[int, int]] = None,
                             normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> List[np.ndarray]:
    """
    Applies remote ANN segmentation model (served by TensorFlow Serving) to a list of images in a series of API
    requests, one request for each batch. Each image is cut into chunks according to chunk_size parameter, then each
    chunk will be resized to model_input_size. All chunks of a batch will be smushed together and sent to TFS to
    process.

    :param images: List of images (np.ndarray) to process
    :param model_name: ANN model name on TFS
    :param model_version: ANN model version
    :param endpoint_url: TFS rest api endpoint URL
    :param batch_size: number of images smushed in a single query to ANN
    :param chunk_size: size of a chunk each image would be cut into. If passed None will be set to (256, 256).
    :param model_input_size: size of a model input (2D). If None it will be computed from ANN metadata
    :param normalization: Normalization function to apply when image is converted into tensor
    :param probabilities_class: Index of class to extract probability mask from
    :return: Results of ANN processing, list of images (np.ndarray)
    """

    if images is None or len(images) == 0:
        return []

    results = []

    pulse_logger = logging.getLogger('irym_tfs_connector.pulse')
    flow_logger = logging.getLogger('irym_tfs_connector.flow')

    if chunk_size is None:
        chunk_size = (256, 256)
        flow_logger.warning('Chunk size parameter received: None. Default size was set = (256, 256)')

    if normalization is None:
        flow_logger.warning('Normalization function parameter received: None. No additional normalization will be '
                            'applied. Make sure you pass normalized data. You may receive wrong results, '
                            'border artifacts and other imperfections if normalization wasn\'t applied correctly')

    if model_input_size is None:
        model_input_size = get_model_input_size_from_metadata(endpoint_url, model_name, model_version)
        flow_logger.warning(f'Model Input Size parameter received: None. Model Input Size was received from model '
                            f'metadata: {model_input_size}.')

        if not model_input_size == (-1, -1) and not model_input_size == chunk_size:
            flow_logger.warning(f'Model Input Size {model_input_size} is not equal to Chunk Size {chunk_size}. Each '
                                f'chunk will be resized to Model Input Size.')

        if model_input_size == (-1, -1):
            flow_logger.warning(f'Model Input Size {model_input_size} means that chunks will be resized by ANN.')

    batches_bounds = __get_bounds_of_batches_by_size(images, batch_size)

    images = pad_images(images, chunk_size)

    batches_count = len(batches_bounds)
    pulse_logger.debug(f'batch_size: {batch_size}, {len(batches_bounds)} batches')

    model_predict_url = get_model_predict_url(endpoint_url, model_name, model_version)

    for idx, bounds in enumerate(batches_bounds):
        pulse_logger.debug(f'Applying model to batch #{idx + 1} of {batches_count}. '
                           f'Length of a batch: {bounds[1] - bounds[0] + 1}.')
        started = datetime.datetime.utcnow()
        stack_results = __apply_segmentation_model_to_batch(images,
                                                            bounds,
                                                            model_predict_url,
                                                            chunk_size,
                                                            model_input_size,
                                                            normalization=normalization)
        results.extend(stack_results)
        ended = datetime.datetime.utcnow()
        pulse_logger.debug(f'Batch #{idx + 1} of {batches_count} took {str(ended - started)}')

    return results


def __apply_model_with_order(thread_index: int, *args) -> Tuple[int, List[np.ndarray]]:
    """
    Proxy function that runs apply_segmentation_model on passed arguments to use in a separate thread. Is used to
    track batches of results from different threads and restore sorting afterwards

    :param thread_index: index of a thread
    :param args: arguments set that will be passed into apply_segmentation_model
    :return: List of resulting images grouped by thread index
    """
    return thread_index, apply_segmentation_model(*args)


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
def __apply_segmentation_model_to_batch(all_images: List[np.ndarray],
                                        batch_bounds: Tuple[int, int],
                                        model_predict_url: str,
                                        chunk_size: Tuple[int, int],
                                        model_input_size: Tuple[int, int],
                                        normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) \
        -> List[np.ndarray]:
    """
    Applies remote TensorFlow Serving model to a batch of images

    :param all_images: Full list of images (to avoid extra copying)
    :param batch_bounds: Bounds of a batch, zero-based, inclusive. (0, 2) will pass items 0, 1 and 2.
    :param model_predict_url: TensorFlow Serving REST API Predict method of a chosen model
    :param chunk_size: Size of a chunk each image in batch will be cut into
    :param model_input_size: Model input size each chunk will be resized to
    :param normalization: Normalization function to apply when image is converted into tensor
    :return: List of resulting images
    """
    batch_thickness = batch_bounds[1] - batch_bounds[0] + 1
    original_image_size = (all_images[0].shape[0], all_images[0].shape[1])

    streamlined_chunked_batch = __create_tfs_input_for_batch(all_images,
                                                             batch_bounds,
                                                             chunk_size,
                                                             model_input_size,
                                                             normalization=normalization)

    predictions = tfl.get_tensorflow_prediction(streamlined_chunked_batch, model_predict_url)

    linear_mask = __calculate_mask_patches_for_tfs_predictions(predictions, chunk_size)

    pathology_maps = __combine_masks_into_pathology_maps(linear_mask,
                                                         batch_thickness, original_image_size,
                                                         chunk_size)

    return pathology_maps


def __calculate_mask_patches_for_tfs_predictions(predictions: list, chunk_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Calculates masks from TensorFlow predictions, resizes them and returns as a linear list of ndarrays
    :param predictions: A list of TensorFlow predictions
    :param chunk_size: Size of a chunk mask will be resized to
    :return: Linear list of masks made from TensorFlow predictions
    """
    linear_tfserving_response = []
    for prediction in predictions:
        mask_patch = np.array(prediction)
        resized_mask_patch = resize_image(mask_patch, chunk_size)
        linear_tfserving_response.append(resized_mask_patch)
    return linear_tfserving_response


@log_metrics('irym_tfs_connector.metrics')
def __create_tfs_input_for_batch(all_images: List[np.ndarray], batch_bounds: Tuple[int, int],
                                 chunk_size: Tuple[int, int],
                                 model_input_size: Tuple[int, int],
                                 normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) \
        -> List[np.ndarray]:
    """
    Creates an input (list of tensors) for a TensorFlow from a chunk of list of (padded) images
    :param all_images: Full list of images (to avoid extra copying)
    :param batch_bounds: Bounds of a batch, zero-based, inclusive. (0, 2) will pass items 0, 1 and 2.
    :param chunk_size: Size of a chunk each image in batch will be cut into
    :param model_input_size: Model input size each chunk will be resized to
    :param normalization: Normalization function to apply when image is converted into tensor
    :return: A list of tensors size of model_input_size ready to be served into TensorFlow
    """
    result = []
    for idx in range(batch_bounds[0], batch_bounds[1] + 1):
        chunked_image: List[np.ndarray] = cut_image_into_chunks(all_images[idx],
                                                                chunk_size,
                                                                model_input_size,
                                                                normalization=normalization)
        result.extend(chunked_image)
    return result


@log_metrics('irym_tfs_connector.metrics')
def __sort_results(result: List[Tuple[int, List[np.ndarray]]]) -> None:
    """
    Sorts processing results by process number
    :param result: List of tuples (process number, list of resulting images)
    """
    result.sort(key=lambda x: x[0])


def __create_mask_default(prediction: list) -> np.ndarray:
    """
    Calculates a mask of multichannel prediction using argmax function
    :param prediction: TensorFlow prediction
    :return: single channel mask image
    """
    prediction = np.argmax(prediction, -1)
    prediction = prediction.astype(np.uint16)
    prediction = prediction[..., np.newaxis]
    squeezed = np.squeeze(prediction, axis=2)
    return squeezed


@log_metrics('irym_tfs_connector.metrics')
def __combine_masks_into_pathology_maps(linear_mask: List[np.ndarray],
                                        batch_thickness: int,
                                        original_image_size: Tuple[int, int],
                                        chunk_size: Tuple[int, int]) -> List[np.ndarray]:
    """
    Combines linear mask of TensorFlow response into a list of images
    :param linear_mask: Linear mask of TensorFlow response
    :param batch_thickness: A number of images in mask
    :param original_image_size: Size of original image
    :param chunk_size: Size of a chunk request was cut into
    :return: A list of single-channel images (pathology maps)
    """

    # TODO по-моему в этот метод передаётся как-то много размеров и толщин, кажется часть из них можно просто
    #  посчитать на месте

    classes_count = linear_mask[0].shape[2]
    chunk_shape = (chunk_size[0], chunk_size[1], classes_count)
    pathology_map_shape = (original_image_size[0], original_image_size[1], classes_count)

    request_map_shape = (original_image_size[0] // chunk_size[0], original_image_size[1] // chunk_size[1])
    result = []
    for idx in range(0, batch_thickness):

        pathology_map = np.zeros(pathology_map_shape)
        patch_start = idx * request_map_shape[0] * request_map_shape[1]
        patch_end = (idx + 1) * request_map_shape[0] * request_map_shape[1]

        patchwork = np.reshape(linear_mask[patch_start:patch_end],
                               (request_map_shape[0], request_map_shape[1], classes_count, -1))

        for x in range(request_map_shape[0]):
            for y in range(request_map_shape[1]):
                pathology_map[(x * chunk_size[0]): (x + 1) * chunk_size[0],
                y * chunk_size[1]: (y + 1) * chunk_size[1]] = np.reshape(patchwork[x][y], chunk_shape)
        result.append(pathology_map)

    return result
