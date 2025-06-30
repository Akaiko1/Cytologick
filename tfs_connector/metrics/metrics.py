import inspect
import logging
from datetime import datetime
from functools import wraps


def log_metrics(logger_name: str):
    """
    Decorator that logs time metrics of function execution
    :param logger_name: A name of a logger to be logged in. When in doubt pass __name__.
    """
    def actual_log_metrics_decorator(func):
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        if not logger.propagate or logger.level > logging.DEBUG:
            return func

        @wraps(func)
        def log_metrics_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            result = func(*args, **kwargs)
            end_time = datetime.utcnow()
            delta_string = str((end_time - start_time))
            logger.debug(f'Function {inspect.getmodule(func).__name__}.{func.__name__} took {delta_string}')
            return result

        return log_metrics_wrapper

    return actual_log_metrics_decorator
