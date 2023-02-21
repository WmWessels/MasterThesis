from typing import Iterable
import functools
import multiprocessing

def batch(iterable: Iterable , batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]
    
def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try: 
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return 0      
            finally:
                pool.close()        
        return inner
    return decorator

class ImputationError(Exception):
    pass