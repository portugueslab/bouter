import warnings
import functools
import inspect

import flammkuchen as fl
from numpy import VisibleDeprecationWarning

from bouter import descriptors


def cache_results(method):
    """ Method decorator that caches an .h5 file with the results of the
    decorated function. This behavior can be disabled with the exp.cache_active
    flag.
    Function results are loaded if the new call arguments match the old call,
    or if the exp.default_cached flag is set.
    :param method:
    :return:
    """

    @functools.wraps(method)
    def decorated_method(exp, **kwargs):
        _, _, keywords, defaults = inspect.getargspec(method)
        if exp.cache_active:
            method_nm = method.__name__

            filename = exp.root / descriptors.CACHE_FILE_TEMPLATE.format(
                exp.session_id, method_nm
            )

            if method_nm in exp.processing_params.keys():
                if (
                    kwargs == exp.processing_params[method_nm]
                    or exp.default_cached
                ):
                    print(
                        f"Using cached {method_nm} (this print will be removed)"
                    )
                    return fl.load(filename)

        # Apply method:
        print(method_nm, kwargs)
        results = method(exp, **kwargs)

        if exp.cache_active:
            fl.save(filename, results)
            exp.update_processing_params({method_nm: kwargs})

        return results

    return decorated_method


def deprecated(message=None):
    if message is None:
        message = ""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                "Function {0} is deprecated. {1}".format(fn.__name__, message),
                category=VisibleDeprecationWarning,
                stacklevel=2,
            )

            return fn(*args, **kwargs)

        return wrapper

    return decorator
