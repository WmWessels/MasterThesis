from sklearn.model_selection import cross_validate

from typing import List, Optional
import openml
import pandas as pd
import signal

import os
import random
import multiprocessing  
import functools
import time
import logging

from pathlib import Path

from gama.utilities.preprocessing import select_categorical_columns, basic_encoding

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, make_scorer, roc_auc_score
import category_encoders as ce

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
    Binarizer,
)
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import (
    SelectFwe,
    SelectPercentile,
    f_classif,
    VarianceThreshold,
)
from sklearn.impute import SimpleImputer

NUMBER_CV = 10

# input: list of pipelines
# list of openml datasets
# for every data set, we get the data, and run every pipeline on it
# we return a list of scores for every data set

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout!")

def force_get_dataset(dataset_id=None, *args, **kwargs):
    """ Remove any existing local files about `dataset_id` and then download new copies. """
    did_cache_dir = openml.utils._create_cache_directory_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, dataset_id, )
    openml.utils._remove_cache_dir_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, did_cache_dir)
    return openml.datasets.get_dataset(dataset_id, *args, **kwargs)

def get_pipelines(path: Path) -> List[Pipeline]:
    """
    Returns a list of pipelines to be run
    """
    pipelines = pd.read_csv(path)
    pipelines_list = pipelines["pipelines"].apply(lambda x: eval(x)).to_list()
    return pipelines_list

def get_datasets(path: Path) -> List[int]:
    """
    Returns a list of openml datasets to be run
    """
    datasets = pd.read_csv(path)
    datasets_list = datasets["did"].to_list()
    return datasets_list

def evaluate_pipeline(pipeline, x_enc, y, cv, scoring, ind, dataset):
    try:
        # log_loss_with_labels = 
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        results = cross_validate(pipeline, x_enc, y, cv = kfold, scoring = scoring, n_jobs = -1, pre_dispatch= "1*n_jobs")
        score = np.mean(results["test_score"])
        time = np.mean(results["fit_time"])
        logging.info(f"{dataset}, {ind}, {score} in {time} seconds")

    except Exception as e:
        print(f"Error on dataset {dataset}, Error: {e}")
        score = None

def run_pipelines(pipelines, datasets, scoring, cv: Optional[int] = 10, is_classification: Optional[bool] = True):
    total_scores = []
    for dataset in datasets:
        dataset_scores = []
        openml_dataset = force_get_dataset(dataset)
        X, y, _, _ = openml_dataset.get_data(
            dataset_format = "dataframe", target = openml_dataset.default_target_attribute
        )
        if pd.api.types.is_sparse(X.iloc[:, 0]):
            X = X.sparse.to_dense()
        x_enc, _ = basic_encoding(X, is_classification=is_classification)

        for ind, pipeline in enumerate(pipelines):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(900)
            try:
                score = evaluate_pipeline(pipeline, x_enc, y, cv, scoring, ind, dataset)
            except TimeoutError:
                score = None
            dataset_scores.append(score)
        total_scores.append(dataset_scores)

    return total_scores

def save_scores(scores, pipelines, datasets, path: Path):
    df = pd.DataFrame(scores, columns = pipelines, index = datasets)
    df.to_csv(path)

def custom_log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred, labels = y_true)

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Get all pipelines
    path_pipelines = Path(__file__).parent / "candidate_pipelines.csv"
    path_dataset = Path(__file__).parent / "multiclass_dids.csv"
    pipelines = get_pipelines(path_pipelines)
    
    # Get all datasets
    datasets = get_datasets(path_dataset)
    # Run all pipelines on all datasets
    scorer = make_scorer(log_loss, greater_is_better = False, needs_proba =True)
    scores = run_pipelines(pipelines, datasets, scoring = scorer)
    # Save scores
    save_path = Path(__file__).parent / "scores.csv"
    save_scores(scores, pipelines, datasets, save_path)

if __name__ == "__main__":
    main()






