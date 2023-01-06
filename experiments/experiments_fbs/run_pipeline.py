from typing import List, Iterator, Tuple, Optional
import openml
import numpy as np
import pandas as pd

import random
import multiprocessing  
import functools
import time
from joblib import Parallel, delayed
from utility import batch

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
        return inner
    return decorator

class PipelineExecutor:

    def select_categorical_columns(
    self,
    df: pd.DataFrame,
    min_f: Optional[int] = None,
    max_f: Optional[int] = None,
    ignore_nan: bool = True,
) -> Iterator[str]:
    
        for column in df.columns:
            if isinstance(df[column].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
                nfactors = df[column].nunique(dropna=ignore_nan)
                if (min_f is None or min_f <= nfactors) and (
                    max_f is None or nfactors <= max_f
                ):
                    yield column

    def basic_encoding(
        self,
        x: pd.DataFrame, is_classification: bool
    ) -> Tuple[pd.DataFrame, TransformerMixin]:

        ord_features = list(self.select_categorical_columns(x, max_f=2))
        if is_classification:
            ord_features.extend(self.select_categorical_columns(x, min_f=11))
        leq_10_features = list(self.select_categorical_columns(x, min_f=3, max_f=10))

        encoding_steps = [
            ("ord-enc", ce.OrdinalEncoder(cols=ord_features, drop_invariant=True)),
            ("oh-enc", ce.OneHotEncoder(cols=leq_10_features, handle_missing="value")),
        ]
        encoding_pipeline = Pipeline(encoding_steps)
        x_enc = encoding_pipeline.fit_transform(x, y=None)  # Is this dangerous?
        return x_enc, encoding_pipeline

    @with_timeout(15)
    def evaluate_pipeline(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> int:
        """
        Evaluates a sklearn pipeline
        ----------
        Parameters

        pipeline: sklearn Pipeline object
        X: Pandas dataframe with the feature matrix
        y: labels
        return_dict: Multiprocessing manager dictionary
        -------
        Returns
        Accuracy of the pipeline on the given data set
        """
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
            pipeline.fit(X_train, y_train)
            label_predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, label_predictions)
        except:
            accuracy = 0
        
        
        return accuracy

    def evaluate_batch(self, indexes: List[int], pipelines: List[Pipeline]) -> pd.DataFrame:
        """
        Evaluates a batch of data sets and pipelines
        ----------
        Parameters
        
        indexes: list of openml data set indexes
        pipelines: list of sklearn pipelines
        -------
        Returns
        A matrix with the size of the index and pipeline lists
        """
        evaluations = []
        for ind in indexes:
            #get dataset from openml
            dataset = openml.datasets.get_dataset(ind)
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
            
            x_enc, _ = self.basic_encoding(X, is_classification=True)
            # manager = multiprocessing.Manager()
            # return_dict = manager.dict()
            # jobs = []
            # for pipeline in pipelines:
                # p = multiprocessing.Process(target = self.evaluate_pipeline, args = (pipeline, x_enc, y, return_dict))
                # jobs.append(p)
                # p.start()
            # for proc in jobs:
                # proc.join(5)
                
            res = Parallel(n_jobs=-1)(delayed(self.evaluate_pipeline)(pipeline, x_enc, y) for pipeline in pipelines)
            evaluations.append(list(res))  
            print(res)
            
        return pd.DataFrame(index = indexes, data = evaluations, columns = [str(pipe) for pipe in pipelines])


def main() -> None:
    random.seed(0)
    data_directory = "../data"
    candidate_pipelines = random.sample(pd.read_csv(f"{data_directory}/candidate_pipelines.csv", skiprows = 1, index_col=0).iloc[:, 0].apply(lambda x: eval(x)).to_list(), 150)
    pipelines_df = pd.DataFrame(candidate_pipelines)
    pipelines_df.to_csv(f"{data_directory}/pipelines.csv")
    indexes = pd.read_csv("../data/good_indexes.csv", index_col = 0).iloc[:, 0]
    pipeline_runner = PipelineExecutor()
    counter = 0
    starting_time = time.time()
    for index_batch in batch(indexes, 50):
        counter += 1
        batch_dataframe = pipeline_runner.evaluate_batch(index_batch, candidate_pipelines)    
        batch_dataframe.to_csv(f"{data_directory}/batch_results/batch_{counter}.csv")
    ending_time = time.time()
    print(f"Evaluated {len(candidate_pipelines)} pipelines on {len(indexes)} data sets, This took {ending_time - starting_time}")

if __name__=="__main__":
    main()



