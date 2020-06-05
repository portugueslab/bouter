import warnings
import functools
import inspect

import flammkuchen as fl
from numpy import VisibleDeprecationWarning

from bouter import descriptors


def get_method_default_kwargs(method):
    argnames, _, _, defaults = inspect.getargspec(method)
    argnames.pop(argnames.index("self"))
    return {n: v for n, v in zip(argnames, defaults)}


def cache_results(target_logfile=None):
    """ Method decorator that caches an .h5 file with the results of the
    decorated function. This behavior can be disabled with the exp.cache_active
    flag.
    Function results are loaded if the new call arguments match the old call,
    or if the exp.default_cached flag is set.
    :param method:
    :param targetfile: if not None, the cached value is written to a target (e.g. a log)
    :return:
    """

    def actual_decorator(wrapped):
        @functools.wraps(wrapped)
        def decorated_method(exp, **kwargs):
            # Combine default parameters and keyword specified arguments:
            full_params_dict = get_method_default_kwargs(wrapped)
            full_params_dict.update(kwargs)

            if exp.cache_active:
                method_nm = wrapped.__name__

                if target_logfile is None:
                    targetfile = (
                        exp.root
                        / descriptors.CACHE_FILE_TEMPLATE.format(
                            exp.session_id, method_nm
                        )
                    )
                else:
                    targetfile = exp._log_filename(target_logfile)
                    print(targetfile)

                if method_nm in exp.processing_params.keys():
                    if full_params_dict == exp.processing_params[method_nm]:
                        print(
                            f"Using cached {method_nm} (this print will be removed)"
                        )
                        return fl.load(targetfile)

            # Apply method:
            results = wrapped(exp, **full_params_dict)

            if exp.cache_active:
                fl.save(targetfile, results)
                exp.update_processing_params({method_nm: full_params_dict})

        return decorated_method

    return actual_decorator


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
