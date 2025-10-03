# core/decorators.py
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        dt = (time.time() - t0) * 1000
        print(f"[timeit] {func.__name__} took {dt:.1f} ms")
        return out
    return wrapper
