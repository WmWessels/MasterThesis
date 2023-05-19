import concurrent.futures
import numpy as np
import pandas as pd
import scipy
import time

from typing import Optional, Any
from itertools import permutations
from pymfe.info_theory import MFEInfoTheory
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score

from utils import impute

class MetaFeatures:
    
    def __init__(
            self,
             X, 
             y,
             is_clf: bool = True,
             n_jobs: Optional[int] = None,
             categorical_indicator: Optional[list] = None
        ):
        if type(X) == pd.DataFrame:
            self.dataframe = X
            self.X = X.values
        elif type(X) == np.ndarray:
            self.X = X
            self.dataframe = pd.DataFrame(X)
        else:
            raise TypeError("X should be a pandas dataframe or numpy array")
        self.y = np.array(y)
        self._njobs = n_jobs
        self.is_clf = is_clf
        self.categorical_indicator = categorical_indicator
        self.meta_features = {}
        self.__post_init__()
        
    def __post_init__(self):
        if self.categorical_indicator is not None:
            self.C = self.X[:, self.categorical_indicator]
            self.N = self.X[:, ~self.categorical_indicator]
        else:
            self.C = self.dataframe.select_dtypes(include=["category"]).values
            self.N = self.dataframe.select_dtypes(np.number).values
        self.landmarking_samples = self.sample_inds(self.N.shape[0])
        self.imputed_N = impute(self.N)
        self.skf, self.attr_folds = self.cv_folds()
    
    def retrieve(self):
        # Calculate generic meta features
        if not self._retrieval_funcs:
            raise ValueError(
                """
                Please set self._retrieval_funcs to the list of meta_features you want to calculate
                before calling retrieve()
                """
            )

        with concurrent.futures.ProcessPoolExecutor(self._njobs) as executor:
            futures = [executor.submit(calculate_feature) for calculate_feature in self._retrieval_funcs]
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
        if not np.any(self.C):
            return {"attr_conc_mean": np.nan, "attr_conc_sd": np.nan}
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

    def attr_entropy(self):
        return np.apply_along_axis(func1d=self.calc_entropy, axis=0, arr=self.C)
    
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

    def calc_entropy(
        self,
        values
    ) -> float:

        _, value_freqs = np.unique(values, return_counts=True)

        return scipy.stats.entropy(value_freqs, base=2)

    @staticmethod
    def sample_inds(
            num_inst: int,
            lm_sample_frac: float = 0.5,
    ) -> np.array:

        ind = np.random.choice(
            a=num_inst, 
            size=int(lm_sample_frac * num_inst), 
            replace=False
        )       

        return ind
    
    def cv_folds(
        self,
        n_splits: Optional[int] = 10
    ):
        if self.is_clf:
            skf = StratifiedKFold(
                n_splits=n_splits
            )
        else:
            skf = KFold(
                n_splits=n_splits
            )
        
        attr_folds = []
        for inds_train, inds_test in skf.split(self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]):
            clf = DecisionTreeRegressor(
            ).fit(self.imputed_N[inds_train, :], y[inds_train])

            attr_folds.append(np.argsort(clf.feature_importances_))

        return skf, np.array(attr_folds, dtype = int)


    @staticmethod
    def calc_joint_ent(
        vec_x: np.ndarray, 
        vec_y: np.ndarray, 
        epsilon: float = 1.0e-8
    ) -> float:
        
        joint_prob_mat = (
            pd.crosstab(vec_y, vec_x, normalize=True).values + epsilon
        )

        joint_ent = np.sum(
            np.multiply(joint_prob_mat, np.log2(joint_prob_mat))
        )

        return -1.0 * joint_ent
    

class ClassificationMetaFeatures(MetaFeatures):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score: Optional[callable] = accuracy_score

    def fit(self):
        self._retrieval_funcs = class_methods(self)

    def _nr_class(self):
        return {"nr_class": len(np.unique(self.y))}

    def _freq_class(self):
        _, counts = np.unique(self.y, return_counts=True)
        freq = counts / self.y.size
        return {"class_freq_mean": np.nanmean(freq), "class_freq_sd": np.nanstd(freq)}

    def _class_conc(self):
        if not np.any(self.C):
            return {"class_conc_mean": np.nan, "class_conc_sd": np.nan}
        
        class_conc = np.apply_along_axis(
            func1d=self.calc_conc, axis=0, arr=self.C, vec_y=self.y
        )
        # class_conc = MFEInfoTheory.ft_class_conc(self.C, self.y)
        return {"class_conc_mean": np.nanmean(class_conc), "class_conc_sd": np.nanstd(class_conc)}
    
    
    def _entropy_attrs(self):
        if not np.any(self.C):
            return {"attr_ent_mean": np.nan, "attr_ent_sd": np.nan, "eq_num_attr_mean": np.nan}
    
        class_ent = self.calc_entropy(self.y)
    
        attr_ent = self.attr_entropy()

        joint_ent = np.apply_along_axis(
                func1d= self.calc_joint_ent, axis=0, arr=self.C, vec_y=self.y
            )

        mut_inf = attr_ent + class_ent - joint_ent

        _, num_col = self.C.shape

        eq_num_attr = float(num_col * class_ent / np.sum(mut_inf))

        return {"attr_ent_mean": np.nanmean(attr_ent), "attr_ent_sd": np.nanstd(attr_ent), "eq_num_attr": eq_num_attr}
    
    
    def _joint_ent(self):
        if not np.any(self.C):
            return {"joint_ent_mean": np.nan, "joint_ent_sd": np.nan}
        joint_ent = MFEInfoTheory.ft_joint_ent(self.C, self.y)
        return {"joint_ent_mean": np.nanmean(joint_ent), "joint_ent_sd": np.nanstd(joint_ent)}

    
    def _best_node(self):
        
        model = DecisionTreeClassifier(
            max_depth=1
        )

        res = np.zeros(self.skf.n_splits, dtype=float)

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"best_node_mean": np.nanmean(res), "best_node_sd": np.nanstd(res)}

    def _linear_discr(self):

        model = LinearDiscriminantAnalysis()

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]
        res = np.zeros(self.skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"linear_discr_mean": np.nanmean(res), "linear_discr_sd": np.nanstd(res)}

    
    def _naive_bayes(self):

        model = GaussianNB()

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        res = np.zeros(self.skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"naive_bayes_mean": np.nanmean(res), "naive_bayes_sd": np.nanstd(res)}

    
    def _random_node(self):

        model =DecisionTreeClassifier(
            max_depth=1
        )

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        rand_ind_attr = np.random.randint(0, N.shape[1], size=1)

        res = np.zeros(self.skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, rand_ind_attr, np.newaxis]
            X_test = N[inds_test, rand_ind_attr, np.newaxis]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"random_node_mean": np.nanmean(res), "random_node_sd": np.nanstd(res)}

    
    def _worst_node(self):

        model = DecisionTreeClassifier(
            max_depth=1
        )

        res = np.zeros(self.skf.n_splits, dtype=float)

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            
            X_train = N[inds_train, self.attr_folds[ind_fold, 0], np.newaxis]
            X_test = N[inds_test, self.attr_folds[ind_fold, 0], np.newaxis]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"worst_node_mean": np.nanmean(res), "worst_node_sd": np.nanstd(res)}

    
class RegressionMetaFeatures(MetaFeatures):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score: Optional[callable] = r2_score

    def fit(self):
        self._retrieval_funcs = class_methods(self)

    def _kurtosis_target(self):
        return {"kurtosis_target": scipy.stats.kurtosis(self.y)}
    
    def _skewness_target(self):
        return {"skewness_target": scipy.stats.skew(self.y)}
    

    def _sd_targ(self):
        return {"sd_target": np.nanstd(self.y)}
    
    def _target_corr(self):
        corr = self.dataframe.corrwith(pd.Series(self.y))
        return {"target_corr_mean": np.nanmean(corr), "target_corr_sd": np.nanstd(corr)}
    
    def _attr_entropy(self):
        if not np.any(self.C):
            return {"attr_entropy_mean": np.nan, "attr_entropy_sd": np.nan}
        attr_entropy = self.attr_entropy()
        return {"attr_entropy_mean": np.nanmean(attr_entropy), "attr_entropy_sd": np.nanstd(attr_entropy)}
    
    def _best_node(self):
        
        model = DecisionTreeRegressor(
            max_depth=1
        )

        res = np.zeros(self.skf.n_splits, dtype=float)

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"best_node_mean": np.nanmean(res), "best_node_sd": np.nanstd(res)}


    def _random_node(self):

        model =DecisionTreeRegressor(max_depth=1)

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        rand_ind_attr = np.random.randint(0, N.shape[1], size=1)

        res = np.zeros(self.skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, rand_ind_attr, np.newaxis]
            X_test = N[inds_test, rand_ind_attr, np.newaxis]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"random_node_mean": np.nanmean(res), "random_node_sd": np.nanstd(res)}

    
    def _worst_node(self):

        model = DecisionTreeRegressor(max_depth=1)

        res = np.zeros(self.skf.n_splits, dtype=float)

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            
            X_train = N[inds_train, self.attr_folds[ind_fold, 0], np.newaxis]
            X_test = N[inds_test, self.attr_folds[ind_fold, 0], np.newaxis]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"worst_node_mean": np.nanmean(res), "worst_node_sd": np.nanstd(res)}
    
    def _linear_regr(self):
        model = LinearRegression()

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]
        res = np.zeros(self.skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"linear_regr_mean": np.nanmean(res), "linear_regr_sd": np.nanstd(res)}
    
    def _bayesian_ridge(self):
        model = BayesianRidge()

        N, y = self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]

        res = np.zeros(self.skf.n_splits, dtype=float)

        for ind_fold, (inds_train, inds_test) in enumerate(self.skf.split(N, y)):
            X_train = N[inds_train, :]
            X_test = N[inds_test, :]
            y_train, y_test = y[inds_train], y[inds_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            res[ind_fold] = self.score(y_test, y_pred)

        return {"bayesian_ridge_mean": np.nanmean(res), "bayesian_ridge_sd": np.nanstd(res)}
    
def class_methods(obj: Any,
                  exclude_pattern: str = "__",
                  include_pattern: str = "_"
                 ) -> list[callable]:

    return [getattr(obj, method) for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith(exclude_pattern) and method.startswith(include_pattern)]

if __name__ == "__main__":
    import openml

    dataset = openml.datasets.get_dataset(2)
    n_jobs = 3

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )
    print(type(X))
    start_time = time.time()
    metafeatures = ClassificationMetaFeatures(X, y,  n_jobs = n_jobs, categorical_indicator = np.array(categorical_indicator))
    metafeatures.fit()
    mf = metafeatures.retrieve()
    print(mf)
    print("--- %s seconds ---" % (time.time() - start_time))
