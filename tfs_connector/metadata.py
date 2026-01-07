from typing import Union, Tuple

import requests

from tfs_connector.metrics import log_metrics
from tfs_connector.model_urls import get_model_metadata_url


@log_metrics('tfs_connector.metrics')
def get_model_metadata(endpoint_url: str, model_name: str, model_version: Union[str, int] = 1) -> dict:
    """
    Gets metadata of a model from TensorFlow REST API
    :param endpoint_url: TensorFlow Serving REST API Url
    :param model_name: Name of a model
    :param model_version: Version of a model
    :return:
    """
    metadata_url = get_model_metadata_url(endpoint_url, model_name, model_version)
    response = requests.get(metadata_url).json()['metadata']
    return response


@log_metrics('tfs_connector.metrics')
def get_model_input_size_from_metadata(endpoint_url: str, model_name: str, model_version: Union[str, int]) \
        -> Tuple[int, int]:
    """
    Gets model input size from TensorFlow Serving REST API
    :param endpoint_url: TensorFlow Serving REST API Url
    :param model_name: Name of a model
    :param model_version: Version of a model
    :return: Model input size (x, y)
    """
    model_metadata = get_model_metadata(endpoint_url, model_name, model_version)
    inputs = model_metadata['signature_def']['signature_def']['serving_default']['inputs']
    first_input_key = next(iter(inputs))

    tensor_shape = inputs[first_input_key]['tensor_shape']
    return int(tensor_shape['dim'][1]['size']), int(tensor_shape['dim'][2]['size'])


@log_metrics('tfs_connector.metrics')
def get_number_of_classes_for_segmentation_model(endpoint_url: str, model_name: str, model_version: Union[str, int]) \
        -> int:
    model_metadata = get_model_metadata(endpoint_url, model_name, model_version)
    inputs = model_metadata['signature_def']['signature_def']['serving_default']['inputs']
    first_input_key = next(iter(inputs))

    tensor_shape = inputs[first_input_key]['tensor_shape']
    return int(tensor_shape['dim'][-1]['size'])
