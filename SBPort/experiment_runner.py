from meta_features import ClassificationMetaFeatures, MetaFeatures
from training.fit_inference_pipeline import PortfolioTransformer
import skops.io as sio

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
import logging
import time
from sklearn.model_selection import cross_val_score
from enum import Enum
from typing import Callable
from utils import prepare_openml_for_inf
from pathlib import Path

import openml
import json
# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.model_selection import cross_val_score, cross_validate

from typing import List, Iterator, Tuple, Optional
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
import json
import time

from meta_features import ClassificationMetaFeatures
import skops.io as sio
from training.fit_inference_pipeline import PortfolioTransformer
from config import inference_kwargs
from inference import SBPort, Task
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import argparse

def one_nn_portfolio(
    task,
    portfolio_size,
    dataset_id
):
    perf_matrix = pd.read_csv(f"training/performance_matrices/performance_matrix_{task}.csv", index_col = 0)
    metafeatures_df = pd.read_csv(f"training/raw_metafeatures/metafeatures_{task}.csv", index_col = 0)
    metafeatures_df = metafeatures_df.applymap(lambda x: np.nan if x == np.inf else x)

    #this is arbitrary. Since all pipelines for a given task are preprocessed in the same way, we only need the first step of the pipeline
    inference_pipeline = sio.load(f"inference_pipelines/bin/inference_pipeline_{task}_kernel_kmeans_3_psize_4", trusted = True)
    preprocessor = inference_pipeline.steps[0][1]

    kwargs = inference_kwargs[f"{task}_kwargs"]
    extractor = kwargs["extractor"]
    numerical_features_with_outliers = kwargs["numerical_features_with_outliers"]
    fit_kwargs = kwargs["fit_kwargs"]

    X, y, categorical_indicator = prepare_openml_for_inf(dataset_id)
    metafeatures_df = preprocessor.transform(metafeatures_df)
    metafeatures_new = SBPort.calculate_metafeatures(
        X, 
        y, 
        extractor = extractor, 
        numerical_features_with_outliers = numerical_features_with_outliers, 
        categorical_indicator = categorical_indicator,
        **fit_kwargs
        )
    metafeatures_new = preprocessor.transform(metafeatures_new)[0]
    
    distances = np.linalg.norm(metafeatures_df - metafeatures_new, axis = 1)
    closest_data_point = np.argmin(distances)

    portfolio = list(
        perf_matrix
        .loc[perf_matrix.index[closest_data_point]]
        .sort_values(ascending = False)[:portfolio_size].index
        )
    return portfolio


if __name__ == "__main__":

    logging.basicConfig(filename = "grid_search.log", level = logging.INFO)
    sbport = SBPort()
    all_datasets = openml.datasets.list_datasets()

    automl_dids = openml.study.get_suite(271).data
    binary_automlbench_dids = [did for did in automl_dids if (all_datasets[did]["NumberOfClasses"] == 2 and all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
    multi_automlbench_dids = [did for did in automl_dids if (all_datasets[did]["NumberOfClasses"] > 2 and all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
    binary_validation, binary_test = train_test_split(binary_automlbench_dids, train_size = 10, random_state = 42)
    multi_validation, multi_test = train_test_split(multi_automlbench_dids, train_size = 10, random_state = 42)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", type = str, choices = ["grid_search", "one_nn", "static"])
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"])
    args = argparser.parse_args()
    task = args.task
    kwargs = inference_kwargs[f"{task}_kwargs"]
    if args.experiment == "grid_search":
        CLUSTER_SIZES = [3, 5, 8]
        PORTFOLIO_SIZES = [4, 8, 16]
        results_dict = {c_size: {p_size: [] for p_size in PORTFOLIO_SIZES} for c_size in CLUSTER_SIZES}
        for dataset_id in binary_validation[1:]:
            for cluster_size in CLUSTER_SIZES:
                inference_pipeline_path = Path(__file__).parent / "inference_pipelines" / str(task) / f"inference_pipeline_{task}_kernel_kmeans_{cluster_size}_psize_{max(PORTFOLIO_SIZES)}"
                result = sbport.run(
                    dataset_id = dataset_id,
                    extractor = kwargs["extractor"],
                    numerical_features_with_outliers = kwargs["numerical_features_with_outliers"],
                    inference_pipeline_path = inference_pipeline_path,
                    **kwargs["fit_kwargs"]
                )
                logging.info(f"result on {cluster_size}: {result}")
                for p_size in PORTFOLIO_SIZES:
                    results_dict[cluster_size][p_size] = results_dict[cluster_size][p_size] + [np.sum(result[:p_size])/p_size]
            results_optics = {p_size: [] for p_size in PORTFOLIO_SIZES}
            # inference_pipeline_path = Path(__file__).parent / "inference_pipelines" / str(task) / f"inference_pipeline_{task}_optics_psize_{max(PORTFOLIO_SIZES)}"
            # result = sbport.run(
            #     dataset_id = dataset_id,
            #     extractor = kwargs["extractor"],
            #     numerical_features_with_outliers = kwargs["numerical_features_with_outliers"],
            #     inference_pipeline_path = inference_pipeline_path,
            #     **kwargs["fit_kwargs"]
            # )
            for p_size in PORTFOLIO_SIZES:
                results_optics[p_size] = results_optics[p_size] + [np.sum(result[:p_size])/p_size]
        # pd.DataFrame(results_dict).applymap(lambda x: sum(x)/len(x)).to_csv(f"grid_search_optics_{task}.csv")
        pd.DataFrame(results_dict).applymap(lambda x: sum(x)/len(x)).to_csv(f"grid_search_{task}.csv")
                
