from functools import wraps
from time import perf_counter
from typing import Callable


def timeit(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = fn(*args, **kwargs)
        print(f"{fn}\tElapsed: {perf_counter() - start: .4f}s")
        return res

    return wrapper
