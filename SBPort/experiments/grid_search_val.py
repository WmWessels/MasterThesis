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

from SBPort.meta_features import ClassificationMetaFeatures
# import skops.io as sio
import skops.io as sio
from training.fit_inference_pipeline import PortfolioTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from utils import prepare_openml_for_inf

def main() -> None:
    
    for dataset_id in binary_validation:

        # dataset = openml.datasets.get_dataset(dataset_id)
        # X, y, categorical_indicator, attribute_names = dataset.get_data(
        #     dataset_format="array", target=dataset.default_target_attribute
        # )
        # X = pd.DataFrame(X, columns=attribute_names)
        # ignore_attributes = dataset.ignore_attribute[0].split(",") if dataset.ignore_attribute else []
        # column_indexer = X.columns.get_indexer_for(ignore_attributes)
        # categorical_indicator = [i for j, i in enumerate(categorical_indicator) if j not in column_indexer]
        # X = X[X.columns[~X.columns.isin(ignore_attributes)]]
        X, y, categorical_indicator = prepare_openml_for_inf(dataset_id)

        meta_feature_extractor = ClassificationMetaFeatures(X, y, n_jobs = 4, is_clf = True, is_binary = True, training = True,categorical_indicator = categorical_indicator)
        meta_feature_extractor.fit()
        mf = meta_feature_extractor.retrieve()
        mf_df = pd.DataFrame(mf, index = range(len(mf)))
        numerical_features_with_outliers = [
        "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean", 
        "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
        ]
        numerical_features_norm = list(set(mf_df.columns) - set(numerical_features_with_outliers))
        mf_df = mf_df[[*numerical_features_with_outliers, *numerical_features_norm]]

        for size in N_CLUSTERS:

            with open(f"inference_pipelines/inference_pipeline_bin_kernel_kmeans_{size}_psize_{max(PORTFOLIO_SIZES)}", "rb") as f:
                inference_pipeline: Pipeline = sio.load(f, trusted = True)
        
            print("-------------------------")
            print("-------------------------")
            print("-------------------------")
            warm_starting_pipelines = inference_pipeline.transform(mf_df)

            scores_actual = []
            logging.info("Starting portfolio evaluation procedure")
            begin_time = time.time()
            for pipeline in warm_starting_pipelines:
                exec_pipeline = eval(pipeline)
                try:
                    score = cross_val_score(exec_pipeline, X, y, cv=10, scoring="roc_auc", n_jobs = -1)
                    scores_actual.append(score.mean())
                except Exception as e:
                    logging.info(f"error: {e}")
                    scores_actual.append(0.5)
            ending_time = time.time()
            for p_size in PORTFOLIO_SIZES:
                logging.info(f"{dataset_id}, {size}, {p_size}, {np.sum(scores_actual[:p_size])/p_size}, {ending_time - begin_time}")

if __name__ == "__main__":
    logging.basicConfig(filename = "gridsearch_bin.log", level=logging.INFO)

    all_datasets = openml.datasets.list_datasets()

    automl_dids = openml.study.get_suite(271).data
    binary_automlbench_dids = [did for did in automl_dids if (all_datasets[did]["NumberOfClasses"] == 2 and all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
    multi_automlbench_dids = [did for did in automl_dids if (all_datasets[did]["NumberOfClasses"] > 2 and all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
    binary_validation, binary_test = train_test_split(binary_automlbench_dids, test_size = 10, random_state = 0)
    multi_validation, multi_test = train_test_split(multi_automlbench_dids, test_size = 10, random_state = 0)

    N_CLUSTERS = [3, 5, 8]
    PORTFOLIO_SIZES = [4, 8, 16]
    main()

