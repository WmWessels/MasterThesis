import openml
from meta_features import MetaFeatures, ClassificationMetaFeatures, RegressionMetaFeatures
from sklearn.metrics import log_loss, make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split

CLUSTER_SIZES = [3, 5, 8]
PORTFOLIO_SIZES = [4, 8, 16]

inference_kwargs = {
    "bin_kwargs": {
        "extractor": ClassificationMetaFeatures,
        "numerical_features_with_outliers": [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean", 
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
            ],
        "scoring": "roc_auc",
        "fit_kwargs": {
            "is_clf": True,
            "is_binary": True
        }
    },
    "multi_kwargs": {
        "extractor": ClassificationMetaFeatures,
        "numerical_features_with_outliers": [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean",
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
            ],
        "scoring": "neg_log_loss",
        "fit_kwargs": {
            "is_clf": True,
            "is_binary": False
        }
    },
    "regr_kwargs": {
        "extractor": RegressionMetaFeatures,
        "numerical_features_with_outliers": [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean",
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd",
            ],
        "scoring": "neg_root_mean_squared_error",
        "fit_kwargs": {
            "is_clf": False,
            "is_binary": False
        }
    }
}

def get_id_mapper(task: str):
    all_datasets = openml.datasets.list_datasets()
    id_mapper = {}
    if task in ["bin", "multi"]:
        automl_dids = openml.study.get_suite(271).data
        binary_automlbench_dids = [did for did in automl_dids if (all_datasets[did]["NumberOfClasses"] == 2 and all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
        multi_automlbench_dids = [did for did in automl_dids if (all_datasets[did]["NumberOfClasses"] > 2 and all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
        binary_validation, binary_test = train_test_split(binary_automlbench_dids, train_size = 0.33, random_state = 42)
        multi_validation, multi_test = train_test_split(multi_automlbench_dids, train_size = 0.33, random_state = 42)
        id_mapper |= {"bin": {"validation": binary_validation, "test": binary_test}, "multi": {"validation": multi_validation, "test": multi_test}}
        return id_mapper
    
    regr_dids = openml.study.get_suite(269).data
    regr_automlbench_dids = [did for did in regr_dids if (all_datasets[did]["NumberOfInstances"]*all_datasets[did]["NumberOfFeatures"] < 10_000_000 and all_datasets[did]["NumberOfInstances"] < 500_000)]
    regr_validation, regr_test = train_test_split(regr_automlbench_dids, train_size = 0.33, random_state = 42)
    regr_test.remove(3050)
    regr_test.remove(3277)
    id_mapper |= {"regr": {"validation": regr_validation, "test": regr_test}}
    return id_mapper

    







