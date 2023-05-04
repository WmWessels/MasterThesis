import concurrent.futures
import numpy as np
import pandas as pd
import scipy

from typing import Optional, Any
from itertools import permutations


class MetaFeatures:
    
    def __init__(
            self,
             X, 
             y,
             n_jobs: Optional[int] = None,
        ):
        if type(X) == pd.DataFrame:
            self.dataframe = X
            self.X = X.values
        elif type(X) == np.array:
            self.X = X
            self.dataframe = pd.DataFrame(X)
        else:
            raise TypeError("X should be a pandas dataframe or numpy array")
        self.y = y
        self.meta_features = {}
        self.__post__init__()

        self._njobs = n_jobs

    def __post__init__(self):
        categorical_frame = self.dataframe.select_dtypes(include=["category"])
        self.C = categorical_frame.values
        self.N = self.dataframe.select_dtypes(np.number).values
        # self.C = self.categorical_frame.values

    def retrieve(self):
        # Calculate generic meta features
        meta_functions = class_methods(self)
        # Calculate individual meta features in parallel
        with concurrent.futures.ThreadPoolExecutor(self._njobs) as executor:
            futures = [executor.submit(calculate_feature) for calculate_feature in meta_functions]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.meta_features.update(result)

        return self.meta_features
    
    def _missing_values(self):
        missing_per_column = self.dataframe.isnull().sum()/self.dataframe.shape[0]
        # missing_per_column = np.count_nonzero(np.isnan(self.X), axis = 0)/self.X.shape[0]
        return {"missing_mean": np.nanmean(missing_per_column), "missing_sd": np.nanstd(missing_per_column)}

    def _shape_attrs(self):
        shape = self.X.shape
        nr_feat = shape[1]
        nr_inst = shape[0]
        attr_to_inst_ratio = nr_feat / nr_inst
        return {"nr_feat": nr_feat, "nr_inst": nr_inst, "attr_to_inst_ratio": attr_to_inst_ratio}

    def _nr_bin(self):
        """
        From pymfe
        """
        bin_cols = np.apply_along_axis(
                func1d=lambda col: np.unique(col).size == 2, axis=0, arr=self.X
            )

        return {"nr_bin": np.sum(bin_cols)}
    
    def _column_types(self):
        cat_types = self.C.shape[1]
        num_types = self.N.shape[1]
        num_to_cat_ratio = 0 if cat_types == 0 else num_types / cat_types
        return {"nr_cat": cat_types, "nr_num": num_types, "num_to_cat_ratio": num_to_cat_ratio}

    def _interquartile_range(self):
        iqr = scipy.stats.iqr(self.N, axis = 0, nan_policy = "omit")
        return {"iqr_mean": np.nanmean(iqr), "iqr_sd": np.nanstd(iqr)}

    def _correlation(self):
        cor = self.dataframe.corr().values.astype(float)
        return {"cor_mean": np.nanmean(cor), "cor_sd": np.nanstd(cor)}

    def _covariance(self):
        cov = self.dataframe.cov().values.astype(float)
        return {"cov_mean": np.nanmean(cov), "cov_sd": np.nanstd(cov)}

    def _kurtosis(self):
        kurtosis = self.dataframe.kurtosis().values.astype(float)
        return {"kurtosis_mean": np.nanmean(kurtosis), "kurtosis_sd": np.nanstd(kurtosis)}
    
    def _max_stats(self):
        max_ = self.dataframe.max().values.astype(float)

        return {"max_mean": np.nanmean(max_), "max_sd": np.nanstd(max_)}

    def _mean_stats(self):
        mean_ = self.dataframe.mean().values.astype(float)
        return {"mean_mean": np.nanmean(mean_), "mean_sd": np.nanstd(mean_)}

    def _median_stats(self):
        median_ = self.dataframe.median().values.astype(float)
        return {"median_mean": np.nanmean(median_), "median_sd": np.nanstd(median_)}

    def _min_stats(self):
        min_ = self.dataframe.min().values.astype(float)
        return {"min_mean": np.nanmean(min_), "min_sd": np.nanstd(min_)}

    def _sd_stats(self):
        sd = self.dataframe.std().values.astype(float)
        return {"sd_mean": np.nanmean(sd), "sd_sd": np.nanstd(sd)}

    def _skewness_stats(self):
        skewness = self.dataframe.skew().values.astype(float)
        return {"skewness_mean": np.nanmean(skewness), "skewness_sd": np.nanstd(skewness)}

    def _variance_stats(self):
        variance = self.dataframe.var().values.astype(float)
        return {"variance_mean": np.nanmean(variance), "variance_sd": np.nanstd(variance)}

    def _outliers(self, whis: float = 1.5):

        v_min, q_1, q_3, v_max = np.percentile(self.N, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return {"outliers": np.sum(np.logical_or(cut_low > v_min, cut_high < v_max))}
    
    def _attr_concentration(
        self, 
        max_attr_num: Optional[int] = 12,
        random_state: Optional[int] = None,
    ) -> np.array:
        _, num_col = self.C.shape

        col_inds = np.arange(num_col)

        if max_attr_num is not None and num_col > max_attr_num:
            if random_state is not None:
                np.random.seed(random_state)

            col_inds = np.random.choice(
                col_inds, size=max_attr_num, replace=False
            )

        col_permutations = permutations(col_inds, 2)

        attr_conc = np.array(
            [
                self.calc_conc(self.C[:, ind_attr_a], self.C[:, ind_attr_b])
                for ind_attr_a, ind_attr_b in col_permutations
            ]
        )

        return {"attr_conc_mean": np.nanmean(attr_conc), "attr_conc_sd": np.nanstd(attr_conc)}

    def _attr_entropy(self):
        attr_entropy = np.apply_along_axis(func1d=self.calc_entropy, axis=0, arr=self.C)
        return {"attr_entropy_mean": np.nanmean(attr_entropy), "attr_entropy_sd": np.nanstd(attr_entropy)}
    
    @staticmethod
    def calc_conc(
        vec_x: np.ndarray, 
        vec_y: np.ndarray, 
        epsilon: float = 1.0e-8
    ) -> float:

        pij = pd.crosstab(vec_x, vec_y, normalize=True).values + epsilon

        isum = pij.sum(axis=0)
        jsum2 = np.sum(pij.sum(axis=1) ** 2)

        conc = (np.sum(pij ** 2 / isum) - jsum2) / (1.0 - jsum2)

        return conc

    @staticmethod
    def calc_entropy(
        values: np.array
    ) -> float:

        _, value_freqs = np.unique(values, return_counts=True)

        return scipy.stats.entropy(value_freqs, base=2)

    # @staticmethod
    # def _calc_joint_ent(
    #     vec_x: np.ndarray, 
    #     vec_y: np.ndarray, 
    #     epsilon: float = 1.0e-8
    # ) -> float:
        
    #     joint_prob_mat = (
    #         pd.crosstab(vec_y, vec_x, normalize=True).values + epsilon
    #     )

    #     joint_ent = np.sum(
    #         np.multiply(joint_prob_mat, np.log2(joint_prob_mat))
    #     )

    #     return -1.0 * joint_ent
    





class ClassificationMetaFeatures(MetaFeatures):
    def retrieve(self, X, y):
        # Calculate generic meta features
        meta_features = super().retrieve(X)
        
        # Calculate classification-specific meta features
        classification_features = {...}
        meta_features.update(classification_features)
        
        return meta_features

class RegressionMetaFeatures(MetaFeatures):
    def retrieve(self, X, y):
        # Calculate generic meta features
        meta_features = super().retrieve(X)
        
        # Calculate regression-specific meta features
        regression_features = {...}
        meta_features.update(regression_features)
        
        return meta_features
    
def class_methods(obj: Any,
                  exclude_pattern: str = "__",
                  include_pattern: str = "_"
                 ) -> list[callable]:
    return [getattr(obj, method) for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith(exclude_pattern) and method.startswith(include_pattern)]

if __name__ == "__main__":
    import openml
    dataset = openml.datasets.get_dataset(5)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )
    print(y)
    metafeatures = MetaFeatures(X, y)
    print(metafeatures.retrieve())