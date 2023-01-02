from typing import List
import openml
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

    @staticmethod
    def evaluate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> int:
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        pipeline.fit(X_train, y_train)
        label_predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, label_predictions)
        return accuracy

    def evaluate_batch(self, indexes: List[int], pipelines: List[Pipeline]) -> np.array:
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
            for pipeline in pipelines:
                accuracy = self.evaluate_pipeline(pipeline, X, y)
                evaluations.append(accuracy)
        return np.array(evaluations)


def main() -> None:
    candidate_pipelines = pd.read_csv("../data/canidate_pipelines.csv", skiprows = 1, index_col=0).iloc[:, 0].apply(lambda x: eval(x)).to_list()
    pipeline_runner = PipelineExecutor()
    indexes = [2]
    pipeline_runner.evaluate_batch(indexes, candidate_pipelines)

if __name__=="__main__":
    main()



