from meta_features import MetaFeatures, ClassificationMetaFeatures, RegressionMetaFeatures

inference_kwargs = {
    "bin_kwargs": {
        "extractor": ClassificationMetaFeatures,
        "numerical_features_with_outliers": [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean", 
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
            ],
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
        "fit_kwargs": {
            "is_clf": True,
            "is_binary": False
        }
    },
    "reg_kwargs": {
        "extractor": RegressionMetaFeatures,
        "numerical_features_with_outliers": [
            "cov_mean", "cov_sd", "iqr_mean", "iqr_sd", "max_mean", "max_sd", "mean_mean", "mean_sd", "median_mean",
            "median_sd", "min_mean", "min_sd", "sd_mean", "sd_sd", "variance_mean", "variance_sd", "eq_num_attr"
            ],
        "fit_kwargs": {
            "is_clf": False,
            "is_binary": False
        }
    }
}




