from typing import List

import numpy as np
import requests
from orjson import orjson

from tfs_connector.metrics import log_metrics


@log_metrics('irym_tfs_connector.metrics')
def get_tensorflow_prediction(chunks_as_tensors: List[np.ndarray], predict_url: str) -> list:
    """
    Sends a linear list of tensors to the TensorFlow Serving REST API predict method
    :param chunks_as_tensors: A linear list of tensors to process
    :param predict_url: TensorFlow Serving REST API predict method Url
    :return: List of lists with TFS results
    """
    predict_request = __prepare_request_json(chunks_as_tensors)
    response = __post_predict_request(predict_url, predict_request)
    prediction = __parse_response_json(response)
    return prediction


@log_metrics('irym_tfs_connector.metrics')
def __prepare_request_json(tensors: List[np.ndarray]) -> bytes:
    """
    Serializes a list of tensors into TFS API request. Made a separate method for logging and profiling purposes.
    """
    instances = [x.tolist() for x in tensors]
    predict_request = orjson.dumps({'instances': instances}, option=orjson.OPT_SERIALIZE_NUMPY)
    return predict_request


@log_metrics('irym_tfs_connector.metrics')
def __post_predict_request(tf_serving_url: str, predict_request: bytes) -> requests.Response:
    """
    Posts data to a Predict TFS API method. Made a separate method for logging and profiling purposes.
    """
    return requests.post(tf_serving_url, predict_request)


def __unprocessible_enitity_response_error(response: requests.Response):
    message = f'Tensorflow API Service returned unprocessable response: HTTP-code: {response.status_code}'
    if response.content is not None:
        message += f' {response.content}'
    return Exception(message)


def __get_tensorflow_exception(response: requests.Response) -> Exception:
    if response.status_code == 504:
        return TimeoutError(f'Tensorflow API Service returned 504 Timeout response: {response.content}')
    return __unprocessible_enitity_response_error(response)


@log_metrics('irym_tfs_connector.metrics')
def __parse_response_json(response: requests.Response) -> list:
    """
    Deserializes json response containing predictions into list of pixels. Made a separate method for
    logging and profiling purposes.
    """

    if response.ok:
        _content = orjson.loads(response.content)
        if 'predictions' in _content:
            return _content['predictions']
        else:
            raise __unprocessible_enitity_response_error(response)
    else:
        exception = __get_tensorflow_exception(response)
        raise exception
