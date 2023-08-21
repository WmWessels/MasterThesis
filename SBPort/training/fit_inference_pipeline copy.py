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

kkmeans_n_clusters = [3, 5, 8]
portfolio_sizes = [4, 8, 16]
optics_kwargs = {
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

def create_transformer(
    data: pd.DataFrame,
    numerical_features_norm: list[str],
    numerical_features_with_outliers: list[str],
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
            ("num_outliers", numerical_transformer_outliers, numerical_features_with_outliers),
            ("num", numerical_transformer_normal, numerical_features_norm),

        ]
    )
    preprocessor.fit(data)

    return preprocessor

def fit_clustering(
    data: pd.DataFrame,
    clustering_algorithm: ClusterMixin,
    **kwargs
) -> ClusterMixin:

    clustering_algorithm.fit(data, **kwargs)
    return clustering_algorithm


def get_metafeature_dataframe(
    task: Task
) -> pd.DataFrame:  

    data = pd.read_csv(
        Path(__file__).parent / "raw_metafeatures" / f"metafeatures_{task}.csv", index_col = 0
        )
    return data

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

def create_inference_pipeline(
    preprocessor: ColumnTransformer,
    clustering_algorithm: ClusterMixin,
) -> Pipeline:

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("cluster", clustering_algorithm)
        ]
    )

    return pipeline

def read_performance_matrix(
    path: Path
):
    performance_matrix = pd.read_csv(path, index_col = 0)
    return performance_matrix

def build_portfolios(
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

def save_portfolios(
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

def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = argparser.parse_args()
    task = args.task
    ascending = False if task ==  "bin" else True

    metafeature_dataframe = get_metafeature_dataframe(task)
    metafeature_dataframe = metafeature_dataframe.applymap(lambda x: np.nan if x == np.inf else x)

    numerical_features_with_outliers = [
        "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean", 
        "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
        ]
    numerical_features_norm = list(set(metafeature_dataframe.columns) - set(numerical_features_with_outliers))
    if task == "regr":
        numerical_features_with_outliers.remove("eq_num_attr")
    transformer = create_transformer(metafeature_dataframe, numerical_features_norm, numerical_features_with_outliers)
    transformed_metafeatures = transformer.transform(metafeature_dataframe)

    performance_matrix = read_performance_matrix(Path(__file__).parent / "performance_matrices" / f"performance_matrix_{task}.csv")

    #Static
    for portfolio_size in portfolio_sizes:
        portfolios = build_portfolios(labels = None, performance_matrix=performance_matrix, portfolio_size=portfolio_size, static=True, ascending=ascending)
        save_portfolios(portfolios, portfolio_size, clustering_algorithm=None, task=str(task))

    #Kernel K-Means
    for cluster_size in kkmeans_n_clusters:
        gamma = 1 if task in ["bin", "multi"] else 0.5
        clustering_algorithm = KernelKMeans(n_clusters = cluster_size, random_state = 42, kernel = "rbf", gamma = gamma)
        fitted_cluster = fit_clustering(transformed_metafeatures, clustering_algorithm)
        for portfolio_size in portfolio_sizes:
            portfolios = build_portfolios(fitted_cluster.labels_, performance_matrix, portfolio_size, ascending=ascending)
            portfoliotransformer = PortfolioTransformer(fitted_cluster, portfolios)
            inference_pipeline = create_inference_pipeline(transformer, str(task), portfoliotransformer)
            save_pipeline(inference_pipeline, str(task), portfoliotransformer.clustering_algorithm, portfolio_size)

    #OPTICS
    clustering_algorithm = MetaOPTICS(mf_dataframe = transformed_metafeatures, **optics_kwargs[str(task)])
    fitted_cluster = fit_clustering(transformed_metafeatures, clustering_algorithm)
    #Set a threshold for the reachability distance, used to enable prediction
    #This is a workaround, since the reachability distance is not set by default
    #No clear way to predict using OPTICS
    fitted_cluster.set_threshold()

    for portfolio_size in portfolio_sizes:
        portfolios = build_portfolios(fitted_cluster.labels_, performance_matrix, portfolio_size)
        portfoliotransformer = PortfolioTransformer(fitted_cluster, portfolios)
        inference_pipeline = create_inference_pipeline(transformer, str(task), portfoliotransformer)
        save_pipeline(inference_pipeline, str(task), portfoliotransformer.clustering_algorithm, portfolio_size)

if __name__ == "__main__":
    main()
 