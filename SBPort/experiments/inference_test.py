import openml
import json
from sklearnex import patch_sklearn
patch_sklearn()

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

from meta_features import ClassificationMetaFeatures
# import skops.io as sio
import skops.io as sio
from training.fit_inference_pipeline import PortfolioTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

all_datasets = openml.datasets.list_datasets()

automl_dids = openml.study.get_suite(271).data
binary_automlbench_dids = [did for did in automl_dids if all_datasets[did]["NumberOfClasses"] == 2]
multi_automlbench_dids = [did for did in automl_dids if all_datasets[did]["NumberOfClasses"] > 2]
binary_validation, binary_test = train_test_split(binary_automlbench_dids, test_size = 0.75, random_state = 42)
multi_validation, multi_test = train_test_split(multi_automlbench_dids, test_size = 0.75, random_state = 42)
label_scores = {}

def main() -> None:
    
    for dataset_id in binary_validation:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )
        meta_feature_extractor = ClassificationMetaFeatures(X, y, n_jobs = 8, is_clf = True, is_binary = True, training = True,categorical_indicator = np.array(categorical_indicator))
        meta_feature_extractor.fit()
        mf = meta_feature_extractor.retrieve()
        print("meta_features: ", mf)
        label_scores[str(dataset_id)] = {}
        for size in [3, 5, 8]:
            label_scores[str(dataset_id)][str(size)] = {}
            with open(f"inference_pipelines/inference_pipeline_bin_kernel_kmeans_{size}_psize_4", "rb") as f:
                inference_pipeline: Pipeline = sio.load(f, trusted = True)
            mf_df = pd.DataFrame(mf, index = range(len(mf)))
            numerical_features_with_outliers = [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean", 
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
            ]
            numerical_features_norm = list(set(mf_df.columns) - set(numerical_features_with_outliers))
            print("-------------------------")
            print("-------------------------")
            print("-------------------------")
            mf_df = mf_df[[*numerical_features_with_outliers, *numerical_features_norm]]
            warm_starting_pipelines = inference_pipeline.transform(mf_df)
            portfolios = inference_pipeline.named_steps["cluster"].portfolios
            for label in range(size):
                warm_starting_pipelines_other = portfolios[str(label)]
                if warm_starting_pipelines_other == warm_starting_pipelines:
                    continue
                scores = []
                for pipeline in warm_starting_pipelines_other:
                    # print(pipeline)
                    exec_pipeline = eval(pipeline)
                    try:
                        score = cross_val_score(exec_pipeline, X, y, cv=10, scoring="roc_auc", n_jobs = -1)
                    except Exception as e:
                        print("error: ", e)
                        continue
                    scores.append(score.mean())
            
                print("avg score on label", label, np.mean(scores))
                label_scores[str(dataset_id)][str(size)][str(label)] = np.mean(scores)

            scores_actual = []
            for pipeline in warm_starting_pipelines:
                exec_pipeline = eval(pipeline)
                score = cross_val_score(exec_pipeline, X, y, cv=10, scoring="roc_auc", n_jobs = -1)
                scores_actual.append(score.mean())
            label_scores[str(dataset_id)][str(size)]["actual"] = np.sum(scores_actual)/size

            performance = []
            for lbl, perf in label_scores[str(dataset_id)][str(size)].items():
                if not lbl == "actual":
                    performance.append(perf)
            label_scores[str(dataset_id)][str(size)]["difference"] = label_scores[str(dataset_id)][str(size)]["actual"] - np.mean(performance)
            label_scores[str(dataset_id)][str(size)]["difference_max"] = label_scores[str(dataset_id)][str(size)]["actual"] - max(label_scores[str(dataset_id)][str(size)].values())
            with open("results_test_3_31.json", "w") as f:
                json.dump(label_scores, f, indent = 4)
            print("avg difference: ", label_scores[str(dataset_id)][str(size)]["difference"])
            print("difference to max: ", label_scores[str(dataset_id)][str(size)]["difference_max"])

if __name__ == "__main__":
    main()

