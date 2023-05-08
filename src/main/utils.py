from typing import Iterable
import numpy as np
import functools
import multiprocessing
import openml
from sklearn.impute import SimpleImputer

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

def get_openml_data(dataset_id=None, *args, **kwargs):
    """ Remove any existing local files about `dataset_id` and then download new copies. """
    did_cache_dir = openml.utils._create_cache_directory_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, dataset_id, )
    openml.utils._remove_cache_dir_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, did_cache_dir)
    dataset = openml.datasets.get_dataset(dataset_id, *args, **kwargs)
    return dataset.get_data

def handling_error(func):
    def wrapper():
        try:
            return func()
        except:
            return 0
    return wrapper

def impute(N: np.array):
        
    if N is None:
        print("No numerical features found, skipping imputation")
        return

    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    imputed_N = imputer.fit_transform(N)

    return imputed_N



