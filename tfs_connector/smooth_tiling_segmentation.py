"""
# Portions of this code are based on examples from:
https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py

"""
import logging
import platform
from concurrent.futures import ThreadPoolExecutor

import multiprocess.pool
from typing import Callable, Optional, List

import numpy as np
import scipy

from tfs_connector.parallelism_mode import ParallelismMode
from tfs_connector.functions import split_evenly
from tfs_connector.image_manipulation import image_to_tensor
from tfs_connector.metadata import get_model_input_size_from_metadata, get_number_of_classes_for_segmentation_model
from tfs_connector.metrics import log_metrics
from tfs_connector.model_urls import get_model_predict_url
from tfs_connector.tensorflow_layer import get_tensorflow_prediction


@log_metrics('tfs_connector.metrics')
def apply_smooth_tiling_segmentation_model(image: np.ndarray,
                                           model_name: str,
                                           endpoint_url: str,
                                           model_version: int = 1,
                                           chunk_size: Optional[int] = None,
                                           subdivisions: int = 2,
                                           thread_count: int = 1,
                                           parallelism_mode: int = 0,
                                           normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) \
        -> np.ndarray:
    """
    Applies remote ANN segmentation model (served by TensorFlow Serving) to an image. Image is split into multiple
    overlapping square chunks based on chunk_size and number of subdivisions and mirrored 8 times, then sent to an ANN
    in a series of API requests. Received responses smushed together using 2D square splines.

    :param parallelism_mode: parallelism mode. Use ParallelismMode.PARALLELISM_MODE_* constants to fill correctly.
    :param image: Input image we want to process
    :param model_name: ANN model name on TFS
    :param endpoint_url: TFS rest api endpoint URL
    :param model_version: ANN model version
    :param chunk_size: a size (1D) of a square (2D) chunk
    :param subdivisions: a number of subdivision. The greater this number is - the more chunks will be created and
    the better smoothing will be achieved. But increasing this number will lead to greater memory usage, greater
    batch sent to TFS and overall slower work of the function.
    :param thread_count: number of parallel threads to run. If 0 or 1 passed single-threaded version of a function will
     be run
    :param normalization: Normalization function to apply when image is converted into tensor
    :return: Results of ANN processing: [..., classes_count] ndarray
    """
    main_logger = logging.getLogger('tfs_connector.main')

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
        thread_count = 1

    if chunk_size is None:
        chunk_size = get_model_input_size_from_metadata(endpoint_url, model_name, model_version)[0]

    nb_classes = get_number_of_classes_for_segmentation_model(endpoint_url, model_name, model_version)

    model_predict_url = get_model_predict_url(endpoint_url, model_name, model_version)

    result = __predict_img_with_smooth_windowing(image, chunk_size=chunk_size, subdivisions=subdivisions,
                                                 nb_classes=nb_classes, thread_count=thread_count,
                                                 parallelism_mode=parallelism_mode,
                                                 prediction_function=__get_send_to_ann_function(model_predict_url,
                                                                                                normalization))

    return result


def __get_send_to_ann_function(call_predict_url: str,
                               normalization: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> \
        Callable[[np.ndarray], List[np.ndarray]]:
    """
    Creates a function that carries ANN URL and normalization function into predict call
    :param call_predict_url: Full URL of TFS's :predict method
    :param normalization: A normalization function
    :return: A function that sends series of images to TFS and returns predictions
    """

    @log_metrics('tfs_connector.metrics')
    def __send_to_ann(_images: np.ndarray) -> List[np.ndarray]:
        image_tensors = [image_to_tensor(x, normalization) for x in _images]
        predictions = get_tensorflow_prediction(image_tensors, call_predict_url)
        return predictions

    return __send_to_ann


@log_metrics('tfs_connector.metrics')
def __pad_image(img: np.ndarray, chunk_size: int, subdivisions: int) -> np.ndarray:
    """
    Adds borders to img for a "valid" border pattern according to "chunk_size" and
    "subdivisions".
    Image is a np array of shape (x, y, nb_channels).

    :param img: Input image.
    :param chunk_size: 1-d size of a square 2-d chunk.
    :param subdivisions: overlapping coefficient for cutting. The greater it is - the more fine cutting you'll get.
    It's better to use powers of 2 for this to avoid rounding problems.
    :return: Padded image of size perfect for chunking.
    """
    aug = int(round(chunk_size * (1 - 1.0 / subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')

    return ret


@log_metrics('tfs_connector.metrics')
def __create_reflections(image: np.ndarray) -> List[np.ndarray]:
    """
    Duplicate an ndarray (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group

    :param image: Image to scatter into reflections
    :return: A list of 8 reflections of an image
    """
    # noinspection PyListCreation
    reflections = []
    reflections.append(np.array(image))
    reflections.append(np.rot90(np.array(image), axes=(0, 1), k=1))
    reflections.append(np.rot90(np.array(image), axes=(0, 1), k=2))
    reflections.append(np.rot90(np.array(image), axes=(0, 1), k=3))
    image = np.array(image)[:, ::-1]
    reflections.append(np.array(image))
    reflections.append(np.rot90(np.array(image), axes=(0, 1), k=1))
    reflections.append(np.rot90(np.array(image), axes=(0, 1), k=2))
    reflections.append(np.rot90(np.array(image), axes=(0, 1), k=3))
    return reflections


@log_metrics('tfs_connector.metrics')
def __predict_img_with_smooth_windowing(input_img: np.ndarray,
                                        chunk_size: int,
                                        subdivisions: int,
                                        nb_classes: int,
                                        thread_count: int,
                                        parallelism_mode: int,
                                        prediction_function: Callable[[np.ndarray], List[np.ndarray]]) -> np.ndarray:
    """
    Apply the `prediction_function` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/

    :param input_img: Input image
    :param chunk_size: A 1D size of 2D square chunk
    :param subdivisions: A number of subdivisions (overlaps) of image chunks
    :param nb_classes: A number of ANN classes
    :param thread_count: A number of parallel threads to run. If 0 or 1 passed single-threaded version of a function
    will be run
    :param prediction_function: a function to run prediction with
    :return: A merged unpadded ANN results
    """
    pad = __pad_image(input_img, chunk_size, subdivisions)
    reflections = __create_reflections(pad)

    # It would also be possible to allow different (and impure) window functions
    # that might not tile well. Adding their weighting to another matrix could
    # be done to later normalize the predictions correctly by dividing the whole
    # reconstructed thing by this matrix of weightings - to normalize things
    # back from an impure windowing function that would have badly weighted
    # windows.

    # For example, since the U-net of Kaggle's DSTL satellite imagery feature
    # prediction challenge's 3rd place winners use a different window size for
    # the input and output of the neural net's patches predictions, it would be
    # possible to fake a full-size window which would in fact just have a narrow
    # non-zero domain. This may require to augment the `subdivisions` argument
    # to 4 rather than 2.

    reflections_prediction_results = []

    if thread_count == 1:
        reflections_prediction_results = __predict_reflections(reflections, chunk_size,
                                                               subdivisions, nb_classes, prediction_function)
    else:
        reflections_split = [x for x in split_evenly(reflections, thread_count)]

        if parallelism_mode == ParallelismMode.PARALLELISM_MODE_MULTIPROCESS:
            with multiprocess.pool.Pool(len(reflections_split)) as pool:
                mpp_results = [pool.apply_async(__predict_reflections, (
                    reflections_slice, chunk_size, subdivisions, nb_classes, prediction_function)) for
                               reflections_slice in reflections_split]

                for mpp_result in mpp_results:
                    reflections_prediction_results.extend(mpp_result.get())

        elif parallelism_mode == ParallelismMode.PARALLELISM_MODE_THREADPOOL:
            with ThreadPoolExecutor(max_workers=len(reflections_split)) as executor:
                futures = [executor.submit(__predict_reflections, reflections_slice,
                                           chunk_size, subdivisions, nb_classes, prediction_function)
                           for reflections_slice in reflections_split]

                for future in futures:
                    reflections_prediction_results.extend(future.result())

    # Merge after rotations:
    padded_results = __combine_reflections(reflections_prediction_results)

    results = __unpad_image(padded_results, chunk_size, subdivisions)

    results = results[:input_img.shape[0], :input_img.shape[1], :]

    return results


@log_metrics('tfs_connector.metrics')
def __predict_reflections(reflections: List[np.ndarray],
                          chunk_size: int,
                          subdivisions: int,
                          nb_classes: int,
                          prediction_function: Callable[[np.ndarray], List[np.ndarray]]) -> List[np.ndarray]:
    """
    Runs prediction_function for a reflections list. Each image in reflection list will be tiled into chunks,
    which then will be streamlined and sent to ANN with prediction_function.
    :param reflections: A list of reflections
    :param chunk_size: A 1D size of 2D square chunk image will be split into
    :param subdivisions: A number of subdivisions (overlaps) of image chunks
    :param nb_classes: A number of classes in results of prediction_function
    :param prediction_function: A function to predict for images
    :return: List of padded results for each reflection in list
    """
    res = []
    for reflection in reflections:
        tiled_prediction_results = __tile_reflection_and_predict(reflection, chunk_size,
                                                                 subdivisions, nb_classes, prediction_function)
        one_padded_result = __combine_result_from_tiles(
            tiled_prediction_results, chunk_size, subdivisions,
            padded_out_shape=list(reflection.shape[:-1]) + [nb_classes])

        res.append(one_padded_result)

    return res


@log_metrics('tfs_connector.metrics')
def __unpad_image(padded_img: np.ndarray, chunk_size: int, subdivisions: int) -> np.ndarray:
    """
    Undo what's done in the `__pad_image` function.
    Image is an ndarray of shape (x, y, nb_channels).
    :param padded_img: A padded image
    :param chunk_size: 1D size of 2D chunk
    :param subdivisions: A number of subdivisions (overlaps) between chunks
    :return: An unpadded image
    """
    aug = int(round(chunk_size * (1 - 1.0 / subdivisions)))
    ret = padded_img[aug:-aug, aug:-aug, :]
    return ret


@log_metrics('tfs_connector.metrics')
def __combine_reflections(reflections: List[np.ndarray]) -> np.ndarray:
    """
    Merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    by the `_rotate_mirror_do` function. Each image might have changed and
    merging them implies to rotate them back in order and average things out.
    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    :param reflections: a list of image reflections
    :return: A combined image.
    """
    combined: List[np.ndarray] = [np.array(reflections[0]), np.rot90(np.array(reflections[1]), axes=(0, 1), k=3),
                                  np.rot90(np.array(reflections[2]), axes=(0, 1), k=2),
                                  np.rot90(np.array(reflections[3]), axes=(0, 1), k=1),
                                  np.array(reflections[4])[:, ::-1],
                                  np.rot90(np.array(reflections[5]), axes=(0, 1), k=3)[:, ::-1],
                                  np.rot90(np.array(reflections[6]), axes=(0, 1), k=2)[:, ::-1],
                                  np.rot90(np.array(reflections[7]), axes=(0, 1), k=1)[:, ::-1]]
    return np.mean(combined, axis=0)


def __combine_result_from_tiles(image_tiles: np.ndarray, chunk_size: int, subdivisions: int,
                                padded_out_shape: List[int]) -> np.ndarray:
    """
    Merge tiled overlapping patches smoothly.
    
    :param image_tiles: Tiles of a image, streamlined into one ndarray
    :param chunk_size: 1D size of 2D chunk
    :param subdivisions: A number of subdivisions (overlaps) between chunks
    :param padded_out_shape: A shape of resulting image.
    :return: An image, combined from chunks
    """
    step = int(chunk_size / subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len - chunk_size + 1, step):
        b = 0
        for j in range(0, pady_len - chunk_size + 1, step):
            windowed_patch = image_tiles[a, b]
            y[i:i + chunk_size, j:j + chunk_size] = y[i:i + chunk_size, j:j + chunk_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def __tile_reflection_and_predict(reflection: np.ndarray,
                                  chunk_size: int,
                                  subdivisions: int,
                                  nb_classes: int,
                                  predict_function: Callable[[np.ndarray], List[np.ndarray]]) -> np.ndarray:
    """
    Create tiled overlapping patches from a reflection and passes them into predict_function.

    :param reflection:  A reflection, streamlined ndarray of 8 image rotations
    :param chunk_size: 1D-size of a square 2D chunk
    :param subdivisions: A number of subdivisions (overlaps) between chunks
    :param nb_classes: A number of classes in results of prediction_function
    :param predict_function: A function to predict
    :return: Tiled prediction results for a chosen reflection
    """
    tiled_reflection = __tile_reflection(reflection, subdivisions, chunk_size)

    a, b, c, d, e = tiled_reflection.shape
    tiled_reflection = tiled_reflection.reshape(a * b, c, d, e)

    processed_reflection = predict_function(tiled_reflection)

    window_spline_2d = _get_2d_spline_window(window_size=chunk_size, power=2)

    processed_reflection = np.array([patch * window_spline_2d for patch in processed_reflection])

    # Such 5D array:
    processed_reflection = processed_reflection.reshape(a, b, c, d, nb_classes)

    return processed_reflection


@log_metrics('tfs_connector.metrics')
def __tile_reflection(reflection: np.ndarray, subdivisions: int, chunk_size: int) -> np.ndarray:
    """
    Cuts a reflection into tiles (chunks) and streamlines them into single ndarray
    :param reflection: A reflection (image) to cut
    :param subdivisions: A number of subdivisions (overlaps) between chunks
    :param chunk_size: 1D-size of a square 2D chunk
    :return: Streamlined into a single ndarray set of overlapping tiles size of chunk_size
    """
    step = int(chunk_size / subdivisions)
    padx_len = reflection.shape[0]
    pady_len = reflection.shape[1]
    result = []
    for i in range(0, padx_len - chunk_size + 1, step):
        result.append([])
        for j in range(0, pady_len - chunk_size + 1, step):
            patch = reflection[i:i + chunk_size, j:j + chunk_size, :]
            result[-1].append(patch)
    result = np.array(result)
    return result


__cached_2d_windows = dict()


def _get_2d_spline_window(window_size: int, power: int = 2) -> object:
    """
    Gets a 2D spline window for set window_size and power. Uses memoization to not calculate splines each time.
    :param window_size: A 1D size of a square 2D window
    :param power: a power of a spline. I'm not sure how it works, so just leave it 2.
    :return: A spline window
    """
    # Memoization
    global __cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in __cached_2d_windows:
        wind = __cached_2d_windows[key]
    else:
        wind = __create_spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)
        wind = wind * wind.transpose(1, 0, 2)
        __cached_2d_windows[key] = wind
    return wind


def __create_spline_window(window_size: int, power: int = 2) -> np.ndarray:
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    :param window_size: a 1D size for a square 2D spline window
    :param power: a power of a spline. I'm not sure how it works, so just leave it 2.
    """
    intersection = int(window_size / 4)
    wind_outer: np.ndarray = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner: np.ndarray = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind: np.ndarray = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind
