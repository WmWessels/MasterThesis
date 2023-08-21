
import os
import logging
import json
import argparse
from pathlib import Path

from typing import Optional

import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
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

from gama import GamaClassifier, GamaRegressor
import skops.io as sio

from config import inference_kwargs, get_id_mapper, CLUSTER_SIZES, PORTFOLIO_SIZES
from inference import SBPort
from utils import prepare_openml_for_inf, sklearn_to_gama_str
from meta_features import ClassificationMetaFeatures, MetaFeatures
from training.candidate_generation import Task
from training.fit_inference_pipeline import PortfolioTransformer

# import autosklearn.classification
# from autosklearn.metrics import roc_auc
# from autosklearn.pipeline.classification import SimpleClassificationPipeline

# from ConfigSpace.configuration_space import Configuration


class ExperimentRunner:

    def __init__(
        self, 
        task: Task, 
        dataset_ids: list[int] | list[str], 
        sbport: SBPort,
        **kwargs
    ):
        self._task = task
        self._dataset_ids = dataset_ids
        self.sbport = sbport
        self.extractor = kwargs["extractor"]
        self.numerical_features_with_outliers = kwargs["numerical_features_with_outliers"]
        self.scoring = kwargs["scoring"]
        self.fit_kwargs = kwargs["fit_kwargs"]

        self.result_dict = {str(dataset_id): {} for dataset_id in self._dataset_ids}
    
    def run_optimal_configuration(
        self,
        inference_pipeline_path: Path,
        search_pattern: str = "max",
    ):
        for file in inference_pipeline_path.iterdir():
            if search_pattern in file.name:
                path = inference_pipeline_path / file.name
        for dataset_id in self._dataset_ids:
            result = self.sbport.run(
                dataset_id = dataset_id,
                extractor = self.extractor,
                numerical_features_with_outliers = self.numerical_features_with_outliers,
                inference_pipeline_path = path,
                scoring = self.scoring,
                **self.fit_kwargs
            )
            logging.info(f"result on {dataset_id}: {result}")
            self.result_dict[str(dataset_id)] = result
    
    def grid_search(
        self,
    ):
        
        for dataset_id in self._dataset_ids:
            for cluster_size in CLUSTER_SIZES:
                inference_pipeline_path = Path(__file__).parent / "inference_pipelines" / str(self._task) / f"inference_pipeline_{self._task}_kernel_kmeans_{cluster_size}_psize_{max(PORTFOLIO_SIZES)}"
                result = self.sbport.run(
                    dataset_id = dataset_id,
                    extractor = self.extractor,
                    numerical_features_with_outliers = self.numerical_features_with_outliers,
                    inference_pipeline_path = inference_pipeline_path,
                    scoring = self.scoring,
                    **self.fit_kwargs
                )
                logging.info(f"result on {cluster_size}: {result}")
                self.result_dict[str(dataset_id)][str(cluster_size)] = result
    
            inference_pipeline_path_optics = Path(__file__).parent / "inference_pipelines" / str(self._task) / f"inference_pipeline_{self._task}_optics_psize_{max(PORTFOLIO_SIZES)}"
            result_optics = self.sbport.run(
                dataset_id = dataset_id,
                extractor = self.extractor,
                numerical_features_with_outliers = self.numerical_features_with_outliers,
                inference_pipeline_path = inference_pipeline_path_optics,
                scoring = self.scoring,
                **self.fit_kwargs
            )
            self.result_dict[str(dataset_id)]["optics"] = result_optics
            logging.info(f"result on optics: {result_optics}")
    
    def one_nn(
        self,
        portfolio_size: int
    ) -> list[list[float]]:
        for dataset_id in self._dataset_ids:
            portfolio, X, y = self._one_nn_portfolio(dataset_id, portfolio_size = portfolio_size)
            score = self.sbport.evaluate_sklearn_pipelines(X, y, portfolio, self.scoring)
            logging.info(f"result on {dataset_id}: {score}")
            self.result_dict[str(dataset_id)] = score

    def _one_nn_portfolio(
        self,
        dataset_id,
        portfolio_size
    ):
        perf_matrix = pd.read_csv(f"training/performance_matrices/performance_matrix_{self._task}.csv", index_col = 0)
        metafeatures_df = pd.read_csv(f"training/raw_metafeatures/metafeatures_{self._task}.csv", index_col = 0)
        metafeatures_df = metafeatures_df.applymap(lambda x: np.nan if x == np.inf else x)

        #this is arbitrary. Since all pipelines for a given task are preprocessed in the same way, we only need the first step of the pipeline
        inference_pipeline = sio.load(f"inference_pipelines/{self._task}/inference_pipeline_{self._task}_kernel_kmeans_3_psize_4", trusted = True)
        preprocessor = inference_pipeline.steps[0][1]

        kwargs = inference_kwargs[f"{self._task}_kwargs"]
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

        portfolio_str = list(
            perf_matrix
            .loc[perf_matrix.index[closest_data_point]]
            .sort_values(ascending = False)[:portfolio_size].index
            )
        portfolio = [eval(pipeline) for pipeline in portfolio_str]
        return portfolio, X, y
    
    def evaluate_static(
        self,
        portfolio_size: int
    ):
        for dataset_id in self._dataset_ids:
            X, y, _ = prepare_openml_for_inf(dataset_id)
            with open(f"inference_pipelines/{self._task}/portfolios_{self._task}_static_psize_{portfolio_size}.json", "rb") as f:
                portfolio = json.load(f)
            portfolio = [eval(pipeline) for pipeline in portfolio]
            score = self.sbport.evaluate_sklearn_pipelines(X, y, portfolio, self.scoring)
            logging.info(f"result on {dataset_id}: {score}")
            self.result_dict[str(dataset_id)] = score

    def run_gama(
        self,
        max_total_time: int = 3600,
        max_eval_time: int = 360,
        store: Optional[str] = "logs",
        warm_start: bool = None,
        warm_start_path: Optional[str] = None,
        n_jobs: int = -1,
    ):  
        for dataset_id in self._dataset_ids:
            X, y, categorical_indicator = prepare_openml_for_inf(dataset_id)
            if warm_start:
                if not warm_start_path:
                    raise ValueError("warm_start_path must be specified if warm_start is set to True")

                metafeatures = self.sbport.calculate_metafeatures(
                    X, 
                    y, 
                    extractor = self.extractor,
                    numerical_features_with_outliers = self.numerical_features_with_outliers,
                    categorical_indicator = categorical_indicator,
                    **self.fit_kwargs
                    )
                inference_pipeline = self.sbport.load_inference_pipeline(warm_start_path)
                warm_start = self.sbport._transform(inference_pipeline, metafeatures)
                if self._task == "regr":
                    warm_start = [sklearn_to_gama_str(pipeline, task = "regression") for pipeline in warm_start]
                else:
                    warm_start = [sklearn_to_gama_str(pipeline) for pipeline in warm_start]

                
            warm_start_postfix = "_ws" if warm_start else ""
            if self._task == "regr":
                clf = GamaRegressor(
                    max_total_time = max_total_time,
                    max_eval_time = max_eval_time,
                    store = store,
                    scoring = self.scoring,
                    n_jobs = n_jobs,
                    output_directory = Path(__file__).parent / f"gama_logs_{dataset_id}_{max_total_time/60}{warm_start_postfix}"
                    )
            else:
                clf = GamaClassifier(
                    max_total_time = max_total_time, 
                    max_eval_time = max_eval_time,
                    store = store, 
                    scoring = self.scoring, 
                    n_jobs = n_jobs,
                    output_directory = Path(__file__).parent / f"gama_logs_{dataset_id}_{max_total_time/60}{warm_start_postfix}"
                )
            print("starting GAMA fit procedure on dataset: ", dataset_id)
            clf.fit(X, y, warm_start = warm_start)

    def run_cluster_vs_all_configuration(
            self,
            inference_pipeline_path: Path,
            search_pattern: str = "max"
        ):
        for file in inference_pipeline_path.iterdir():
            if search_pattern in file.name:
                path = inference_pipeline_path / file.name
        results = {dataset_id: {} for dataset_id in self._dataset_ids}
        for dataset_id in self._dataset_ids:
            X, y, categorical_indicator = prepare_openml_for_inf(dataset_id)
            metafeatures = self.sbport.calculate_metafeatures(
                    X, 
                    y, 
                    extractor = self.extractor,
                    numerical_features_with_outliers = self.numerical_features_with_outliers,
                    categorical_indicator = categorical_indicator,
                    **self.fit_kwargs
                    )

            inference_pipeline = self.sbport.load_inference_pipeline(path)
            portfolios = inference_pipeline.named_steps["cluster"].portfolios

            transformed_mf = inference_pipeline.named_steps["preprocessor"].transform(metafeatures)
            predicted_label = inference_pipeline.named_steps["cluster"].clustering_algorithm.predict(transformed_mf)
            logging.info(f"predicted_label: {predicted_label}")

            n_clusters = len(portfolios)
            
            for label in range(n_clusters):
                portfolio = portfolios[str(label)]
                portfolio = [eval(pipeline) for pipeline in portfolio]
                result = self.sbport.evaluate_sklearn_pipelines(X, y, portfolio, self.scoring)
                if label == predicted_label:
                    self.result_dict[str(dataset_id)]["predicted"] = result
                else:
                    self.result_dict[str(dataset_id)][str(label)] = result
                logging.info(f"result on {dataset_id} for cluster {label}: {result}")


        # if not os.path.isdir(Path(__file__).parent / "results" / "cluster_vs_all"):
        #     os.mkdir(Path(__file__).parent / "results" / "cluster_vs_all")
        # with open(Path(__file__).parent / "results" / "cluster_vs_all" / f"cluster_vs_all_{self._task}.json", "w") as f:
        #     json.dump(results, f, indent = 4)

    def run_askl2(
    self,
    portfolio_size: int
    ):
        portfolio_postfix = "_logloss" if self._task == "multi" else "_roc"
        with open(f"experiments/askl2_portfolios/portfolio{portfolio_postfix}.json", "r") as file:
            portfolio = json.load(file)
        for dataset_id in self._dataset_ids:
            X, y, _ = prepare_openml_for_inf(dataset_id)
            y = LabelEncoder().fit_transform(y)

            cv_results = self._run_askl2(X, y, portfolio["portfolio"], portfolio_size) 
            logging.info(f"result on {dataset_id}: {cv_results}")
            self.result_dict[str(dataset_id)] = cv_results

    def _run_askl2(
        self,
        X, 
        y,
        port: dict[str, Configuration],
        portfolio_size: int
    ):
        """
        This function is quite convoluted, but Auto-Sklearn does not provide a convenient utility function to evaluate their portfolios outside of AutoML context.
        Additionally, the pipelines are not standard scikit-learn pipelines. As such, the cross_validation functionality from sklearn cannot directly be used.
        """
        def custom_cross_val_score(estimator, X, y, cv=10, random_state=42, n_jobs=1):
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state = random_state)

            def fit_and_score(train_index, test_index):
                try:
                    X_train, y_train = X[train_index], y[train_index]
                    X_test, y_test = X[test_index], y[test_index]

                    estimator.fit(X_train, y_train)
                    y_pred = estimator.predict_proba(X_test)[:, 1]
                except Exception as e:
                    print(e)
                    return 0
                return roc_auc(y_test, y_pred)

            scores = Parallel(n_jobs=n_jobs)(
                delayed(fit_and_score)(train_index, test_index)
                for train_index, test_index in splitter.split(X, y)
            )
            return scores

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=3600, 
            per_run_time_limit=900,
            delete_tmp_folder_after_terminate=False
            )

        config_space = automl.get_configuration_space(X, y)
        hp_names = config_space.get_hyperparameter_names()

        initial_configurations = []

        for member in port.values():
            _member = {key: member[key] for key in member if key in hp_names}
            initial_configurations.append(
                Configuration(configuration_space=config_space, values=_member)
            )

        cv_scores = []
        for config in initial_configurations[:portfolio_size]:
            pipeline, _, _ = automl.fit_pipeline(
                X = X,
                y = y,
                config = config
            )
            if pipeline:
                score = np.mean(custom_cross_val_score(pipeline, X, y, cv=10, random_state=42, n_jobs=-1))
            else:
                print("Error fitting pipeline")
                score = np.nan
            cv_scores.append(score)
        return cv_scores


def get_portfolio_size_optimal(
    path_to_optimal_dir: Path,
    search_pattern: str = "max"
):
    
    portfolio_size = None

    for file in path_to_optimal_dir.iterdir():
        if search_pattern in file.name:
            portfolio_size = file.name.split("_")[-1]

    if not portfolio_size:
        raise ValueError("No portfolio size found, did you evaluate the optimal configurations?")

    return int(portfolio_size)

def main():
    logging.basicConfig(level = logging.INFO, filename = "experiment_runner.log", filemode = "w")
    sbport = SBPort()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", type = str, choices = ["grid_search", "one_nn", "cluster_vs_all", "static", "optimal_max", "optimal_heur", "askl2", "automl_10", "automl_60", "automl_ws_10", "automl_ws_60"])
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"])
    args = argparser.parse_args()
    task = args.task
    kwargs = inference_kwargs[f"{task}_kwargs"]
    id_mapper = get_id_mapper(task)
    experiment_runner = ExperimentRunner(task, id_mapper[str(task)]["test"], sbport, **kwargs)
    path_to_optimal_dir = Path(__file__).parent / "optimal_configurations" / str(task)
    if args.experiment == "grid_search":
        experiment_runner._dataset_ids = id_mapper[str(task)]["validation"]
        experiment_runner.grid_search()

    elif args.experiment == "one_nn":
        portfolio_size = get_portfolio_size_optimal(path_to_optimal_dir)
        experiment_runner.one_nn(portfolio_size)
    
    elif args.experiment == "cluster_vs_all":
        experiment_runner.run_cluster_vs_all_configuration(path_to_optimal_dir)
    
    elif args.experiment == "static":
        portfolio_size = get_portfolio_size_optimal(path_to_optimal_dir)
        experiment_runner.evaluate_static(portfolio_size)

    elif args.experiment == "optimal_max":
        experiment_runner.run_optimal_configuration(path_to_optimal_dir, search_pattern = "max")
    
    elif args.experiment == "optimal_heur":
        experiment_runner.run_optimal_configuration(path_to_optimal_dir, search_pattern = "heuristic")

    elif args.experiment == "askl2":
        portfolio_size = get_portfolio_size_optimal(path_to_optimal_dir)
        experiment_runner.run_askl2(portfolio_size = portfolio_size)

    elif args.experiment == "automl_10":
        experiment_runner.run_gama(max_total_time = 600, max_eval_time = 360)

    elif args.experiment == "automl_60":
        experiment_runner.run_gama(max_total_time = 3600, max_eval_time = 360)
    
    elif args.experiment == "automl_ws_10":
        for file in path_to_optimal_dir.iterdir():
            if "heuristic" in file.name:
                path = path_to_optimal_dir / file.name

        experiment_runner.run_gama(max_total_time = 600, max_eval_time = 360, warm_start = True, warm_start_path = path)

    elif args.experiment == "automl_ws_60":
        for file in path_to_optimal_dir.iterdir():
            if "heuristic" in file.name:
                path = path_to_optimal_dir / file.name

        experiment_runner.run_gama(max_total_time = 3600, max_eval_time = 360, warm_start = True, warm_start_path = path)

if __name__ == "__main__":
    main()
    
                
