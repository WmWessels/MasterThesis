from typing import Tuple, Callable, Any, Optional
import re
import numpy as np
import functools
import pandas as pd
import openml
from gama.utilities.preprocessing import select_categorical_columns, basic_encoding
import category_encoders as ce
from gama.configuration.classification import clf_config
from gama.configuration.regression import reg_config
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
    dataset_id: int | str,
    task: str = "classification"
) -> Tuple[pd.DataFrame, np.array, np.array]:
    dataset = openml.datasets.get_dataset(dataset_id)
    
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    # if pd.api.types.is_sparse(X.iloc[:, 0]):
    #     X = X.sparse.to_dense()
    # X, _ = basic_encoding(X, is_classification = task == "classification")
    X = pd.DataFrame(X, columns=attribute_names)
    ignore_attributes = dataset.ignore_attribute[0].split(",") if dataset.ignore_attribute else []
    column_indexer = X.columns.get_indexer_for(ignore_attributes)
    categorical_indicator = np.array([i for j, i in enumerate(categorical_indicator) if j not in column_indexer])
    X = X[X.columns[~X.columns.isin(ignore_attributes)]]
    return X, y, categorical_indicator


def sklearn_to_gama_str(
    sklearn_pipe: Pipeline,
    task: str = "classification"
) -> str:
    #Code from https://github.com/openml-labs/gama/issues/156
    #Extended to support more sklearn models
    config = clf_config if task == "classification" else reg_config
    sklearn_pipe = sklearn_pipe[1:] if sklearn_pipe['imputation'] else sklearn_pipe
    l = []
    for i in range(len(sklearn_pipe)):
        l.append(str(sklearn_pipe[i].__class__()).replace('()',''))
#making string from pipeline
    s = []
#For making list
    for i in reversed(l):
        s.append(f"{i}(")
#for making data 
    data_string ="data"
    s.append(data_string)
#for making hyperparameters
    for i in range(len(sklearn_pipe)):

        keys = sklearn_pipe[i].__dict__.keys() & config[sklearn_pipe[i].__class__].keys()
        for j in keys:
        # if j in clf_config[p[i].__class__].keys():
            if j == list(keys)[-1]:
                if type(sklearn_pipe[i].__dict__[j])==str:

                    s.append(f"{str(sklearn_pipe[i].__class__()).replace('()','')}.{j}='{sklearn_pipe[i].__dict__[j]}'")
                else:

                    s.append(f"{str(sklearn_pipe[i].__class__()).replace('()','')}.{j}={sklearn_pipe[i].__dict__[j]}")
            else:
                if type(sklearn_pipe[i].__dict__[j])==str:
                    s.append(f"{str(sklearn_pipe[i].__class__()).replace('()','')}.{j}='{sklearn_pipe[i].__dict__[j]}', ")
                else:
                    s.append(f"{str(sklearn_pipe[i].__class__()).replace('()','')}.{j}={sklearn_pipe[i].__dict__[j]}, ")
        s.append('), ')
    s[-1] = ')'
    s = [item.split(".")[-1] if "min_samples" in item else item for item in s]
    
    s = ''.join(s)
    pattern = r"<(.*?)>"
    if not task == "classification":
        s = re.sub(pattern, "f_regression", s)
    else:
        s = re.sub(pattern, "f_classif", s)
    s = s.replace("dataP", "data, P")
    s = s.replace("dataS", "data, S")
    s = s.replace("datamin", "data, min")
    s = s.replace("(data), )", "(data))")
    s = s.replace("dataF", "data, F")
    s = s.replace("dataG", "data, G")
    s = s.replace("dataE", "data, E")
    s = s.replace("dataR", "data, R")
    s = s.replace("dataN", "data, N")
    s = s.replace("dataB", "data, B")
    s = s.replace("dataV", "data, V")
    s = s.replace("), )", ")")
    s = s.replace("LinearSVR.dual='warn'", "LinearSVR.dual=False")
    return s

def extend_pipeline(X):
    many_factor_features = list(select_categorical_columns(X, min_f = 11))
    encoder = ce.TargetEncoder(cols = many_factor_features)
    return encoder
