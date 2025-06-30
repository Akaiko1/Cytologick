from typing import Union


def get_model_predict_url(endpoint_url: str, model_name: str, model_version: Union[int, str] = 1) -> str:
    """
    Formats given parameters into TensorFlow REST API Predict method Url
    :param endpoint_url: TensorFlow REST API Url
    :param model_name: Name of a model
    :param model_version: Version of a model
    :return: TensorFlow REST API Predict method Url
    """
    if endpoint_url.endswith('/'):
        endpoint_url = endpoint_url.strip('/')
    model_url = f'{endpoint_url}/v1/models/{model_name}/versions/{model_version}:predict'
    return model_url


def get_model_metadata_url(endpoint_url: str, model_name: str, model_version: Union[str, int] = 1) -> str:
    """
    Formats given parameters into TensorFlow REST API Metadata method Url
    :param endpoint_url: TensorFlow Serving REST API Url
    :param model_name: Name of a model
    :param model_version: Version of a model
    :return: TensorFlow REST API Metadata method Url
    """
    if endpoint_url.endswith('/'):
        endpoint_url = endpoint_url.strip('/')
    metadata_url = f'{endpoint_url}/v1/models/{model_name}/versions/{model_version}/metadata'
    return metadata_url
