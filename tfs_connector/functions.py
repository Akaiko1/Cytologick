from typing import List, Generator

from tfs_connector.metrics import log_metrics


@log_metrics('tfs_connector.metrics')
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


