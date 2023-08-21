import numpy as np
import pandas as pd
import json
from pathlib import Path
import skops.io as sio
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer    
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import ClusterMixin
from sklearn.cluster import OPTICS
from sklearn.base import BaseEstimator, TransformerMixin
from clustering import KernelKMeans, MetaOPTICS

from candidate_generation import Task

N_CLUSTERS = [3, 5, 8]
PORTFOLIO_SIZES = [4, 8, 16]
OPTICS_KWARGS = {
    "bin": {
        "eps": 0.5,
        "min_samples": 10,
        "metric": "euclidean",
        "n_jobs": -1,
    },
    "multi": {
        "eps": 0.5,
        "min_samples": 5,
        "xi": 0.001,
        "metric": "euclidean",
        "n_jobs": -1,
    },
    "regr": {
        "eps": 0.5,
        "min_samples": 10,
        "metric": "euclidean",
        "n_jobs": -1,
    }
}



class PortfolioTransformer(BaseEstimator, TransformerMixin):
    """
    This class serves as a tiny wrapper around our clustering algorithms to directly predict portfolios instead of cluster labels.
    """
    def __init__(
        self, 
        clustering_algorithm: ClusterMixin, 
        portfolios: dict[str, list[Pipeline]]
    ):

        self.clustering_algorithm = clustering_algorithm
        self.portfolios = portfolios
    
    def fit(self, X, y = None):
        self.clustering_algorithm.fit(X)
        return self
    
    def transform(self, X, y = None):
        label = self.clustering_algorithm.predict(X)
        portfolio = self.portfolios[str(label[0])]
        return portfolio


def get_metafeature_dataframe(
    task: Task
) -> pd.DataFrame:  

    data = pd.read_csv(
        Path(__file__).parent / "raw_metafeatures" / f"metafeatures_{task}.csv", index_col = 0
        )
    return data





def read_performance_matrix(
    path: Path
):
    performance_matrix = pd.read_csv(path, index_col = 0)
    return performance_matrix




    
class InferencePipeline:

    def __init__(self, task: Task, ascending: bool = True):
        self.task = task
        self._ascending = ascending

        self._metafeature_dataframe = get_metafeature_dataframe(task).applymap(lambda x: np.nan if x == np.inf else x)

        self._numerical_features_with_outliers = [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean", 
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
            ]
        self._numerical_features_norm = list(set(self._metafeature_dataframe.columns) - set(self._numerical_features_with_outliers))
        if self.task == "regr":
            self._numerical_features_with_outliers.remove("eq_num_attr")

    def create_transformer(
        self,
    ) -> ColumnTransformer:
        """
        Transformer function for meta features. Ensures that our data is imputed accordingly and scaled for isotropy.
        Returns a fitted ColumnTransformer object.
        """

        numerical_transformer_normal = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy = "mean")),
                ("scaler", MinMaxScaler())
            ]
        )

        numerical_transformer_outliers = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy = "median")),
                ("scaler", StandardScaler())
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num_outliers", numerical_transformer_outliers, self._numerical_features_with_outliers),
                ("num", numerical_transformer_normal, self._numerical_features_norm),

            ]
        )
        preprocessor.fit(self._metafeature_dataframe)

        return preprocessor
    
    def fit_clustering(
        self,
        data: pd.DataFrame,
        clustering_algorithm: ClusterMixin,
        **kwargs
    ) -> ClusterMixin:

        clustering_algorithm.fit(data, **kwargs)
        return clustering_algorithm
    
    @staticmethod
    def save_pipeline(
        pipeline: Pipeline,
        task: Task,
        clustering_algorithm: ClusterMixin,
        portfolio_size: int,
    ) -> None:
        """
        Creates path referring to the task and the clustering algorithm used and saves the pipeline object to that path.
        """

        if isinstance(clustering_algorithm, OPTICS):
            path_ending = "optics"
        else:
            path_ending = f"kernel_kmeans_{clustering_algorithm.n_clusters}"
        path = Path(__file__).parent.parent / "inference_pipelines" / str(task) / f"inference_pipeline_{task}_{path_ending}_psize_{portfolio_size}"

        sio.dump(pipeline, path)

    
    def construct_portfolio(
        self,
        labels: list[int],
        performance_matrix: pd.DataFrame,
        portfolio_size: int,
        static: bool = False,
        ascending: str = False
    ) -> dict[str, list[Pipeline]]:
        """
        Constructs portfolios using greedy selection
        """
        if static:
            return list(
                performance_matrix
                .mean(axis = 0)
                .sort_values(ascending = ascending)
                [:portfolio_size].index
                )
        portfolios = {}
        for label in set(labels):
            label_ind = np.where(labels == label)
            portfolios[str(label)] = list(
                performance_matrix.loc[performance_matrix.index[label_ind]]
                .mean(axis = 0)
                .sort_values(ascending = ascending)
                [:portfolio_size].index
            )
        return portfolios
    
    def create_inference_pipeline(
        self,
        column_transformer: ColumnTransformer,
        clustering_algorithm: ClusterMixin,
    ) -> Pipeline:

        pipeline = Pipeline(
            steps=[
                ("preprocessor", column_transformer),
                ("cluster", clustering_algorithm)
            ]
        )

        return pipeline
    
    def save_portfolio(
        self,
        portfolios: dict[str, str],
        portfolio_size: int,
        clustering_algorithm: ClusterMixin,
        task: Task,
    ) -> None:  
        """
        Only used for static portfolios. Dynamic portfolios are saved directly in the pipeline object.
        """

        if isinstance(clustering_algorithm, OPTICS):
            path_ending = "optics"
        elif isinstance(clustering_algorithm, KernelKMeans):
            path_ending = f"kernel_kmeans_{clustering_algorithm.n_clusters}"
        else:
            path_ending = "static"
        path = Path(__file__).parent.parent / "inference_pipelines" / str(task) / f"portfolios_{task}_{path_ending}_psize_{portfolio_size}.json"

        with open(path, "w") as f:
            json.dump(
                portfolios, 
                f, 
                indent = 4
            )
    
    def construct_portfolios(
        self
    ) -> None:
        self._column_transformer = self.create_transformer()
        self._preprocessed_metafeatures = self._column_transformer.transform(self._metafeature_dataframe)
        self._performance_matrix = pd.read_csv(Path(__file__).parent / "performance_matrices" / f"performance_matrix_{self.task}.csv", index_col = 0)
        self.construct_static_portfolios()
        gamma = 1 if self.task in ["bin", "multi"] else 0.5
        self.construct_kkmeans_pipelines(gamma = gamma)
        self.construct_optics_pipelines()
    
    def construct_kkmeans_pipelines(
        self,
        gamma: float = 1,
    ):  

        for cluster_size in N_CLUSTERS:
            clustering_algorithm = KernelKMeans(n_clusters = cluster_size, random_state = 42, kernel = "rbf", gamma = gamma)
            fitted_cluster = clustering_algorithm.fit(self._preprocessed_metafeatures)

            for portfolio_size in PORTFOLIO_SIZES:
                portfolios = self.construct_portfolio(
                    labels = fitted_cluster.labels_, 
                    performance_matrix = self._performance_matrix, 
                    portfolio_size = portfolio_size, 
                    ascending = self._ascending
                )

                portfoliotransformer = PortfolioTransformer(fitted_cluster, portfolios)
                inference_pipeline = self.create_inference_pipeline(
                    column_transformer = self._column_transformer, 
                    clustering_algorithm = portfoliotransformer
                )

                self.save_pipeline(
                    pipeline = inference_pipeline, 
                    task = self.task, 
                    clustering_algorithm = portfoliotransformer.clustering_algorithm, 
                    portfolio_size = portfolio_size
                )
        
    def construct_optics_pipelines(
        self
    ) -> None:
        clustering_algorithm = MetaOPTICS(mf_dataframe = self._preprocessed_metafeatures, **OPTICS_KWARGS[self.task])
        fitted_cluster = clustering_algorithm.fit(self._preprocessed_metafeatures)
        #Set a threshold for the reachability distance, used to enable prediction
        #This is a workaround, since the reachability distance is not set by default
        #No clear way to predict using OPTICS
        fitted_cluster.set_threshold()

        for portfolio_size in PORTFOLIO_SIZES:
            portfolios = self.construct_portfolio(
                labels = fitted_cluster.labels_,
                performance_matrix = self._performance_matrix, 
                portfolio_size = portfolio_size
            )

            portfoliotransformer = PortfolioTransformer(fitted_cluster, portfolios)
            inference_pipeline = self.create_inference_pipeline(
                column_transformer = self._column_transformer,
                clustering_algorithm = portfoliotransformer
            )

            self.save_pipeline(
                pipeline = inference_pipeline, 
                task = self.task, 
                clustering_algorithm = portfoliotransformer.clustering_algorithm, 
                portfolio_size = portfolio_size
            )
    
    
    def construct_static_portfolios(
        self
    ) -> None:
        for portfolio_size in PORTFOLIO_SIZES:
            portfolio = self.construct_portfolio(
                labels = None, 
                performance_matrix = self._performance_matrix, 
                portfolio_size = portfolio_size, 
                static = True, 
                ascending = self._ascending
            )
            self.save_portfolio(
                portfolio, 
                portfolio_size, 
                clustering_algorithm = None, 
                task = self.task
            )

def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = argparser.parse_args()
    task = args.task
    ascending = False if task ==  "bin" else True
    inference_pipeline = InferencePipeline(task = task, ascending = ascending)
    inference_pipeline.construct_portfolios()

if __name__ == "__main__":
    main()
 