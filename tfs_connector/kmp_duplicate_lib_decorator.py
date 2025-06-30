import os
from typing import Optional


def save_and_restore_kmp_duplicate_lib_ok(function):
    def wrapper(*args, **kwargs):
        kmp_duplicate_lib_ok = __preserve_kmp_duplicate_lib_ok()
        result = function(*args, **kwargs)
        __restore_kmp_duplicate_lib_ok(kmp_duplicate_lib_ok)
        return result
    return wrapper


def __restore_kmp_duplicate_lib_ok(kmp_duplicate_lib_ok: Optional[str]) -> None:
    """
    Restores value of environment variable KMP_DUPLICATE_LIB_OK
    :param kmp_duplicate_lib_ok: value for KMP_DUPLICATE_LIB_OK env var. If passed None KMP_DUPLICATE_LIB_OK
    will be deleted.
    """
    if kmp_duplicate_lib_ok:
        os.environ["KMP_DUPLICATE_LIB_OK"] = kmp_duplicate_lib_ok
    else:
        del os.environ["KMP_DUPLICATE_LIB_OK"]


def __preserve_kmp_duplicate_lib_ok() -> Optional[str]:
    """
    Saves the value of environment variable KMP_DUPLICATE_LIB_OK. This env var commands libraries to not panic if
    there's several copies of them loaded.
    :return: KMP_DUPLICATE_LIB_OK env var value if set
    """
    old_value = os.environ.get("KMP_DUPLICATE_LIB_OK", None)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return old_value
