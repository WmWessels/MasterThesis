import signal
import logging
import time
from typing import Callable
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics._scorer import _ProbaScorer

from sklearn.model_selection import cross_val_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
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
    VarianceThreshold,
    f_regression,
)

from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.impute import SimpleImputer

import skops.io as sio

from .meta_features import MetaFeatures
from .training.fit_inference_pipeline import PortfolioTransformer
from .utils import prepare_openml_for_inf

def timeout_handler(signum, frame):
    raise TimeoutError("Cross validation procedure for this Pipeline took more than 900 seconds, terminating it.")

signal.signal(signal.SIGALRM, timeout_handler)

class SBPort:

    @staticmethod
    def calculate_metafeatures(
        X: pd.DataFrame, 
        y: pd.Series | np.ndarray, 
        extractor: MetaFeatures,
        numerical_features_with_outliers = list[str],
        categorical_indicator: np.ndarray | None = None,
        **fit_kwargs
    ) -> pd.DataFrame:

        meta_feature_extractor = extractor(X, y, categorical_indicator = categorical_indicator, **fit_kwargs)
        meta_feature_extractor.fit()
        mf = meta_feature_extractor.retrieve()
        
        numerical_features_norm = list(set(mf.keys()) - set(numerical_features_with_outliers))
        columns = [*numerical_features_with_outliers, *numerical_features_norm]

        mf_df = pd.DataFrame(mf, index = [0], columns = columns)

        return mf_df
    
    def _transform(
        self,
        inference_pipeline: Pipeline,
        metafeatures: pd.DataFrame
    ) -> list[Pipeline]:
        """
        Pipelines are stored as strings. This function evaluates them and returns a list of executable sklearn pipelines.
        """

        return [eval(pipeline) for pipeline in inference_pipeline.transform(metafeatures)]

    @staticmethod
    def load_inference_pipeline(
        path: Path
    ):
        with open(path, "rb") as f:
            inference_pipeline: Pipeline = sio.load(f, trusted = True)
        return inference_pipeline

    def evaluate_sklearn_pipelines(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        pipelines: list[str],
        scoring: str | Callable[..., float]
    ) ->  list[float]:
        """
        Function to evaluate a batch of sklearn pipelines
        """

        scores = []
        logging.info("Starting portfolio evaluation procedure")
        begin_time = time.time()
        for pipeline in pipelines:
            score = self._evaluate_sklearn_pipeline(pipeline, X, y, scoring = scoring)
            scores.append(score)
        ending_time = time.time()
        logging.info(f"Portfolio evaluation procedure took {ending_time - begin_time} seconds")
        return scores

    def _evaluate_sklearn_pipeline(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame, 
        y: pd.Series | np.ndarray,
        scoring: str | Callable[..., float]
    ) -> float:
        signal.alarm(900)
        if isinstance(scoring, _ProbaScorer):
            scoring._kwargs.update({"labels": y})
        try:
            score = cross_val_score(pipeline, X, y, cv = 10, scoring = scoring, n_jobs = -1).mean()
        except Exception as e:
            logging.error(f"Error while evaluating pipeline {pipeline}: {e}")
            score = np.nan
        return score
    
    def run(
        self,
        dataset_id: int | str,
        extractor: MetaFeatures,
        numerical_features_with_outliers: list[str],
        inference_pipeline_path: Path,
        scoring: str | Callable[..., float],
        **fit_kwargs
    ) -> list[float]:
        """
        Runs the whole inference pipeline on a new data set and performs detached portfolio evaluation.
        """
        X, y, categorical_indicator = prepare_openml_for_inf(dataset_id)
        metafeatures = self.calculate_metafeatures(
            X, 
            y, 
            extractor = extractor,            
            numerical_features_with_outliers = numerical_features_with_outliers,
            categorical_indicator = categorical_indicator,
            **fit_kwargs
        )
        inference_pipeline = self.load_inference_pipeline(inference_pipeline_path)
        pipelines = self._transform(inference_pipeline, metafeatures)

        scores = self.evaluate_sklearn_pipelines(X, y, pipelines, scoring)
        return scores

