import functools
import inspect
import warnings
from logging import info

import flammkuchen as fl
from numpy import VisibleDeprecationWarning

from bouter import descriptors


def get_method_default_kwargs(method):
    argnames, _, _, defaults, _, _, _ = inspect.getfullargspec(method)
    argnames.pop(argnames.index("self"))
    if len(argnames) > 0:
        return {n: v for n, v in zip(argnames, defaults)}
    else:
        return dict()


def cache_results(cache_filename=None):
    """Method decorator that caches an .h5 file with the results of the
    decorated function. This behavior can be disabled with the exp.cache_active
    flag.
    Function results are loaded if the new call arguments match the old call,
    or if the exp.default_cached flag is set.
    :param method:
    :param cache_filename: if not None, the cached value is written to a target (e.g. a log)
    :return:
    """

    def actual_decorator(wrapped):
        @functools.wraps(wrapped)
        def decorated_method(exp, force_recompute=False, **kwargs):
            # Combine default parameters and keyword specified arguments:
            no_new_paramters = len(kwargs) == 0

            full_params_dict = get_method_default_kwargs(wrapped)
            full_params_dict.update(kwargs)

            # If we are in caching mode:
            if exp.cache_active:
                method_nm = wrapped.__name__  # name of the function

                # produce filename for the cache:
                if cache_filename is None:
                    targetfile = (
                        exp.root
                        / descriptors.CACHE_FILE_TEMPLATE.format(
                            exp.session_id, method_nm
                        )
                    )
                else:
                    targetfile = exp.root / exp._log_filename(cache_filename)

                # If we already produced outputs for the function, we used
                # the same parameters, and we don't force recalculation:
                if (
                    method_nm in exp.processing_params.keys()
                    and (
                        full_params_dict == exp.processing_params[method_nm]
                        or no_new_paramters
                    )
                    and not force_recompute
                ):
                    info(f"Using cached {method_nm} in {targetfile}")
                    return fl.load(targetfile, "/data")

            # Apply the function we are decorating:
            results = wrapped(exp, **full_params_dict)

            # If we are in caching mode, store results
            if exp.cache_active:
                fl.save(targetfile, dict(data=results))
                exp.update_processing_params({method_nm: full_params_dict})

            return results

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
