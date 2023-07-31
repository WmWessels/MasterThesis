from typing import Tuple, Iterable, Callable, Any, Optional
import numpy as np
import functools
import pandas as pd
import openml
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def force_get_dataset(dataset_id=None, *args, **kwargs):
    """ Remove any existing local files about `dataset_id` and then download new copies. """
    did_cache_dir = openml.utils._create_cache_directory_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, dataset_id, )
    openml.utils._remove_cache_dir_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, did_cache_dir)
    return openml.datasets.get_dataset(dataset_id, *args, **kwargs)

def handling_error(
    func: Callable[..., float]
) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            value = func(*args, **kwargs)
        except:
            value = None
        return value
    return wrapper

def impute(N: np.array):
        
    if N is None or N.size == 0:
        print("No numerical features found, skipping imputation")
        return N

    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    imputed_N = imputer.fit_transform(N)

    return imputed_N

def plot_2d_cluster(
    X, 
    reduction_method,
    is_kernel = True,
    labels = list[int],
    random_state: Optional[int] = 0,
    gamma: Optional[float] = 1
    ) -> None:
    if is_kernel:
        X = RBFSampler(gamma = gamma, random_state = random_state).fit_transform(X)

    if reduction_method == "pca":
        reduced_X = PCA(n_components = 2, random_state = random_state).fit_transform(X)
    
    elif reduction_method == "tsne":
        reduced_X = TSNE(n_components = 2, random_state = random_state).fit_transform(X)

    plt.scatter(reduced_X.T[0], reduced_X.T[1], c = labels)

def prepare_openml_for_inf(
    dataset_id: int | str
) -> Tuple[pd.DataFrame, np.array, np.array]:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    X = pd.DataFrame(X, columns=attribute_names)
    ignore_attributes = dataset.ignore_attribute[0].split(",") if dataset.ignore_attribute else []
    column_indexer = X.columns.get_indexer_for(ignore_attributes)
    categorical_indicator = np.array([i for j, i in enumerate(categorical_indicator) if j not in column_indexer])
    X = X[X.columns[~X.columns.isin(ignore_attributes)]]
    return X, y, categorical_indicator



