from sklearn.model_selection import cross_val_score
from .utils import get_openml_data

from typing import List, Iterator, Tuple, Optional
import openml
import pandas as pd

import os
import random
import multiprocessing  
import functools
import time
from joblib import Parallel, delayed
from gama.utilities.preprocessing import select_categorical_columns, basic_encoding

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def run_pipelines(pipelines, datasets, scoring, cv: Optional[int] = 10, is_classification: Optional[bool] = True):
    total_scores = []
    for dataset in datasets:
        dataset_scores = []
        X, y, _, _ = get_openml_data(dataset)
        x_enc, _ = basic_encoding(X, is_classification=is_classification)
        for pipeline in pipelines:
            try:
                score = cross_val_score(pipeline, x_enc, y, cv = cv, scoring = scoring).mean()
            except Exception as e:
                print(f"Error on dataset {dataset}, Error: {e}")
                score = None
            dataset_scores.append(score)
        total_scores.append(dataset_scores)

    return total_scores




