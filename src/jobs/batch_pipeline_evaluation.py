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

from main.utils import batch, with_timeout


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

class PipelineExecutor:

    @with_timeout(15)
    def evaluate_pipeline(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> int:
        """
        Evaluates a sklearn pipeline
        ----------
        Parameters

        pipeline: sklearn Pipeline object
        X: Pandas dataframe with the feature matrix
        y: labels
        -------
        Returns
        Accuracy of the pipeline on the given data set
        """
        
        try:
            #Fit sklearn pipeline. When any kind of error occurs, return 0 as accuracy
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
            pipeline.fit(X_train, y_train)
            label_predictions = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, label_predictions)
        except Exception as e:
            print(e)
            accuracy = 0

        return accuracy

    def evaluate_batch(self, dataset_ids: List[int], pipelines: List[Pipeline]) -> pd.DataFrame:
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
        for dataset_id in dataset_ids:
            dataset = openml.datasets.get_dataset(dataset_id)
            #Try to get the data. 
            #On failure, return a row with 0's for easy processing of the results and not crashing the batch job.
            try:
                X, y, _, _ = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
            except Exception as e:
                print(e) 
                evaluations.append([None for i in range(len(pipelines))])
                continue
            
            x_enc, _ = basic_encoding(X, is_classification=True)
                
            res = Parallel(n_jobs=3)(delayed(self.evaluate_pipeline)(pipeline, x_enc, y) for pipeline in pipelines)
            evaluations.append(list(res))  
            
        return pd.DataFrame(index = dataset_ids, data = evaluations, columns = [str(pipe) for pipe in pipelines])


def main() -> None:
    data_directory = os.path.join(os.getcwd(), "/src/data")
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    # random.seed(0)
    # candidate_pipelines = random.sample(pd.read_csv(f"{data_directory}/candidate_pipelines.csv", skiprows = 1, index_col=0).iloc[:, 0].apply(lambda x: eval(x)).to_list(), 150)
    candidate_pipelines = pd.read_csv(f"{data_directory}/pipelines.csv", skiprows = 1, index_col=0).iloc[:, 0].apply(lambda x: eval(x))
    indexes = [2, 3]
    pipeline_runner = PipelineExecutor()
    batch_size = 20
    starting_time = time.time()
    counter = 0
    for index_batch in batch(indexes, batch_size):
        counter += 1
        batch_dataframe = pipeline_runner.evaluate_batch(index_batch, candidate_pipelines)    
        batch_dataframe.to_csv(f"{data_directory}/batch_results/batch_{counter}.csv")
    ending_time = time.time()
    print(f"""Evaluated {len(candidate_pipelines)} pipelines on {len(indexes)} data sets, 
              This took {ending_time - starting_time}""")

if __name__=="__main__":
    main()



