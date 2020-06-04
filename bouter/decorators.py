import warnings
import functools
import flammkuchen as fl

from numpy import VisibleDeprecationWarning


CACHE_FILE_TEMPLATE = "cache_{}.h5"


def cache_results(method):
    """ Method decorator that caches an .h5 file with the results of the
    decorated function. This behavior can be disabled with the exp.cache_active
    flag.
    Function results are loaded if the new call arguments match the old call,
    or if the exp.default_cached flag is set.
    :param method:
    :return:
    """

    def decorated_method(exp, **kwargs):
        if exp.cache_active:
            # Create cache file if none exists:
            filename = exp.root / CACHE_FILE_TEMPLATE.format(method.__name__)

            if filename.exists():
                old_arguments = fl.load(filename, "/arguments")

                if kwargs == old_arguments or exp.default_cached:
                    print("Using cached ", method.__name__)
                    return fl.load(filename, "/results")

        results = method(exp, **kwargs)

        if exp.cache_active:
            fl.save(filename, dict(results=results, arguments=kwargs))

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
