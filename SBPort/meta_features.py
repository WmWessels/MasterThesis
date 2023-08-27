import time
import logging
import numpy as np
import pandas as pd
import scipy
from pathlib import Path
import json
import openml
import argparse
import multiprocessing
from joblib import Parallel, delayed

from typing import Optional, Any
from itertools import permutations
from pymfe.info_theory import MFEInfoTheory
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score

from utils import impute, handling_error

N_JOBS = multiprocessing.cpu_count() // 2

bin_kwargs = {
    "is_clf": True,
    "is_binary": True,
    "training": True
}

multi_kwargs = {
    "is_clf": True,
    "is_binary": False,
    "training": True
}

regr_kwargs = {
    "is_clf": False,
    "is_binary": False,
    "training": True
}

RANDOM_STATE = 42

class MetaFeatures:
    
    def __init__(
            self,
             X, 
             y,
             is_clf: bool,
             is_binary: bool,
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
        if not n_jobs:
            self._njobs = N_JOBS
        self._njobs = n_jobs
        self.is_clf = is_clf
        self.is_binary = is_binary
        self.categorical_indicator = categorical_indicator

        with open("_features.json", "r") as f:
            meta_features = json.load(f)
        
        if not self.is_clf:
            self.meta_features = meta_features["meta_features_regr"]
        elif self.is_binary:
            self.meta_features = meta_features["meta_features_bin"]
        else:
            self.meta_features = meta_features["meta_features_multi"]

        self.meta_features = {feature: np.nan for feature,_ in self.meta_features.items()}
        
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

        if not self._retrieval_funcs:
            raise ValueError(
                """
                Please set self._retrieval_funcs to the list of meta_features you want to calculate
                before calling retrieve()
                """
            )

        begin_time = time.time()
        results = Parallel(n_jobs = self._njobs)(delayed(calculate_feature)() for calculate_feature in self._retrieval_funcs)
        for result in results:
            if result is not None:
                self.meta_features.update(result)
        
        ending_time = time.time()
        logging.info(f"Calculated metafeatures in {ending_time - begin_time} seconds")

        return self.meta_features
    
    @handling_error
    def _missing_values(self):
        missing_per_column = self.dataframe.isnull().sum()/self.dataframe.shape[0]
        # missing_per_column = np.count_nonzero(np.isnan(self.X), axis = 0)/self.X.shape[0]
        return {"missing_mean": np.nanmean(missing_per_column), "missing_sd": np.nanstd(missing_per_column)}

    @handling_error
    def _shape_attrs(self):
        shape = self.X.shape
        nr_feat = shape[1]
        nr_inst = shape[0]
        attr_to_inst_ratio = nr_feat / nr_inst
        return {"nr_feat": nr_feat, "nr_inst": nr_inst, "attr_to_inst_ratio": attr_to_inst_ratio}
        
    @handling_error
    def _nr_bin(self):
        """
        From pymfe
        """
        bin_cols = np.apply_along_axis(
                func1d=lambda col: np.unique(col).size == 2, axis=0, arr=self.X
            )

        return {"nr_bin": np.sum(bin_cols)}
    
    @handling_error
    def _column_types(self):
        cat_types = self.C.shape[1]
        num_types = self.N.shape[1]
        num_to_cat_ratio = 0 if cat_types == 0 else num_types / cat_types
        return {"nr_cat": cat_types, "nr_num": num_types, "num_to_cat_ratio": num_to_cat_ratio}

    @handling_error
    def _interquartile_range(self):
        iqr = scipy.stats.iqr(self.N, axis = 0, nan_policy = "omit")
        return {"iqr_mean": np.nanmean(iqr), "iqr_sd": np.nanstd(iqr)}

    @handling_error
    def _correlation(self):
        cor = self.dataframe.corr(numeric_only=True).values.astype(float)
        return {"cor_mean": np.nanmean(cor), "cor_sd": np.nanstd(cor)}

    @handling_error
    def _covariance(self):
        cov = self.dataframe.cov().values.astype(float)
        return {"cov_mean": np.nanmean(cov), "cov_sd": np.nanstd(cov)}

    @handling_error
    def _kurtosis(self):
        kurtosis = self.dataframe.kurtosis().values.astype(float)
        return {"kurtosis_mean": np.nanmean(kurtosis), "kurtosis_sd": np.nanstd(kurtosis)}
    
    @handling_error
    def _max_stats(self):
        max_ = self.dataframe.max().values.astype(float)

        return {"max_mean": np.nanmean(max_), "max_sd": np.nanstd(max_)}

    @handling_error
    def _mean_stats(self):
        mean_ = self.dataframe.mean().values.astype(float)
        return {"mean_mean": np.nanmean(mean_), "mean_sd": np.nanstd(mean_)}

    @handling_error
    def _median_stats(self):
        median_ = self.dataframe.median().values.astype(float)
        return {"median_mean": np.nanmean(median_), "median_sd": np.nanstd(median_)}

    @handling_error
    def _min_stats(self):
        min_ = self.dataframe.min().values.astype(float)
        return {"min_mean": np.nanmean(min_), "min_sd": np.nanstd(min_)}

    @handling_error
    def _sd_stats(self):
        sd = self.dataframe.std().values.astype(float)
        return {"sd_mean": np.nanmean(sd), "sd_sd": np.nanstd(sd)}

    @handling_error
    def _skewness_stats(self):
        skewness = self.dataframe.skew().values.astype(float)
        return {"skewness_mean": np.nanmean(skewness), "skewness_sd": np.nanstd(skewness)}

    @handling_error
    def _variance_stats(self):
        variance = self.dataframe.var().values.astype(float)
        return {"variance_mean": np.nanmean(variance), "variance_sd": np.nanstd(variance)}

    @handling_error
    def _outliers(self, whis: float = 1.5):

        v_min, q_1, q_3, v_max = np.percentile(self.N, (0, 25, 75, 100), axis=0)

        whis_iqr = whis * (q_3 - q_1)

        cut_low = q_1 - whis_iqr
        cut_high = q_3 + whis_iqr

        return {"outliers": np.sum(np.logical_or(cut_low > v_min, cut_high < v_max))}

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

        np.random.seed(RANDOM_STATE)

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
        if self.imputed_N is None or self.imputed_N.size == 0:
            return None, None
        if self.is_clf:
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle = True,
                random_state = RANDOM_STATE
            )
        else:
            skf = KFold(
                n_splits=n_splits,
                shuffle = True,
                random_state = RANDOM_STATE
            )
        
        attr_folds = []
        for inds_train, inds_test in skf.split(self.imputed_N[self.landmarking_samples, :], self.y[self.landmarking_samples]):
            clf = DecisionTreeRegressor(
                random_state = RANDOM_STATE
            ).fit(self.imputed_N[inds_train, :], self.y[inds_train])

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
    
    @handling_error
    def _attr_concentration(
        self, 
        max_attr_num: Optional[int] = 12
    ) -> np.array:

        _, num_col = self.C.shape

        col_inds = np.arange(num_col)

        if max_attr_num is not None and num_col > max_attr_num:
            np.random.seed(RANDOM_STATE)

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
        if not attr_conc:
            return

        return {"attr_conc_mean": np.nanmean(attr_conc), "attr_conc_sd": np.nanstd(attr_conc)}
    

class ClassificationMetaFeatures(MetaFeatures):

    def __init__(self, *args, is_binary = True, is_clf = True, **kwargs):
        super().__init__(*args, is_binary = is_binary, is_clf = is_clf, **kwargs)
        self.score: Optional[callable] = accuracy_score
        self.is_binary = is_binary

    def fit(self):
        self._retrieval_funcs = class_methods(self)

    def _class_ftrs(self):
        labels, counts = np.unique(self.y, return_counts=True)
        if self.is_binary:
            return {"majority_class_size": np.max(counts)/self.y.size, "minority_class_size": np.min(counts)/self.y.size}
        else:
            return {"majority_class_size": np.max(counts)/self.y.size, "minority_class_size": np.min(counts)/self.y.size, "nr_class": len(labels)}

    @handling_error
    def _class_conc(self):
       
        class_conc = np.apply_along_axis(
            func1d=self.calc_conc, axis=0, arr=self.C, vec_y=self.y
        )
        
        return {"class_conc_mean": np.nanmean(class_conc), "class_conc_sd": np.nanstd(class_conc)}
    
    @handling_error
    def _entropy_attrs(self):
    
        class_ent = self.calc_entropy(self.y)
    
        attr_ent = self.attr_entropy()

        joint_ent = np.apply_along_axis(
                func1d= self.calc_joint_ent, axis=0, arr=self.C, vec_y=self.y
            )

        mut_inf = attr_ent + class_ent - joint_ent

        _, num_col = self.C.shape

        eq_num_attr = float(num_col * class_ent / np.sum(mut_inf))

        return {"attr_ent_mean": np.nanmean(attr_ent), "attr_ent_sd": np.nanstd(attr_ent), "eq_num_attr": eq_num_attr}
    
    @handling_error
    def _joint_ent(self):
        joint_ent = MFEInfoTheory.ft_joint_ent(self.C, self.y)
        return {"joint_ent_mean": np.nanmean(joint_ent), "joint_ent_sd": np.nanstd(joint_ent)}

    @handling_error
    def _best_node(self):
        
        model = DecisionTreeClassifier(
            max_depth=1,
            random_state = RANDOM_STATE
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

        return {"best_node_mean": np.nanmean(res)}

    @handling_error
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

        return {"linear_discr_mean": np.nanmean(res)}

    @handling_error
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

        return {"naive_bayes_mean": np.nanmean(res)}

    @handling_error
    def _random_node(self):

        model =DecisionTreeClassifier(
            max_depth=1,
            random_state = RANDOM_STATE
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

        return {"random_node_mean": np.nanmean(res)}

    @handling_error
    def _worst_node(self):

        model = DecisionTreeClassifier(
            max_depth=1,
            random_state = RANDOM_STATE
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

        return {"worst_node_mean": np.nanmean(res)}

    
class RegressionMetaFeatures(MetaFeatures):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score: Optional[callable] = r2_score

    def fit(self):
        self._retrieval_funcs = class_methods(self)

    @handling_error
    def _kurtosis_target(self):
        return {"kurtosis_target": scipy.stats.kurtosis(self.y)}
    
    @handling_error
    def _skewness_target(self):
        return {"skewness_target": scipy.stats.skew(self.y)}
    
    @handling_error
    def _sd_targ(self):
        return {"sd_target": np.nanstd(self.y)}
    
    @handling_error
    def _target_corr(self):
        corr = self.dataframe.corrwith(pd.Series(self.y))
        return {"target_corr_mean": np.nanmean(corr), "target_corr_sd": np.nanstd(corr)}
    
    @handling_error
    def _attr_entropy(self):
        attr_entropy = self.attr_entropy()
        if attr_entropy.size == 0:
            return None
        return {"attr_entropy_mean": np.nanmean(attr_entropy), "attr_entropy_sd": np.nanstd(attr_entropy)}

    @handling_error
    def _best_node(self):
        
        model = DecisionTreeRegressor(
            max_depth=1,
            random_state = RANDOM_STATE
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

        return {"best_node_mean": np.nanmean(res)}

    @handling_error
    def _random_node(self):

        model = DecisionTreeRegressor(
            max_depth=1,
            random_state=RANDOM_STATE
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

        return {"random_node_mean": np.nanmean(res)}

    @handling_error
    def _worst_node(self):

        model = DecisionTreeRegressor(
            max_depth=1,
            random_state=RANDOM_STATE
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

        return {"worst_node_mean": np.nanmean(res)}

    @handling_error
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

        return {"linear_regr_mean": np.nanmean(res)}

    @handling_error
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

        return {"bayesian_ridge_mean": np.nanmean(res)}
    
def class_methods(obj: Any,
                  exclude_pattern: str = "__",
                  include_pattern: str = "_"
                 ) -> list[callable]:
    """
    All class methods containing meta features are written as private methods. 
    This function returns all private methods of a class.
    """

    return [getattr(obj, method) for method in dir(obj) if callable(getattr(obj, method)) and not method.startswith(exclude_pattern) and method.startswith(include_pattern)]

def run(
    dataset_id: int,
    extractor: MetaFeatures,
    **kwargs
) -> pd.DataFrame:

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="array",
        target=dataset.default_target_attribute
    )
    if type(X) == scipy.sparse.csr_matrix:
        X = X.toarray()
    mf_extractor = extractor(X, y, categorical_indicator = np.array(categorical_indicator), **kwargs)
    
    
    mf_extractor.fit()
    mf = mf_extractor.retrieve()

    return mf
        
def run_batch(
    dataset_ids: list[int],
    task_type: str,
    extractor: MetaFeatures,
    save: bool = True, 
    **kwargs
) -> list[dict[str, float]]:

    metafeatures_list = []

    for dataset_id in dataset_ids:
        metafeatures = run(dataset_id, extractor, **kwargs)
        metafeatures_list.append(metafeatures)

    metafeatures_dataframe = pd.DataFrame(metafeatures_list, index = dataset_ids)
    if save:
        metafeatures_dataframe.to_csv(f"training/raw_metafeatures/metafeatures_{task_type}.csv")
    
    return metafeatures_dataframe

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task_type", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = argparser.parse_args()
    if args.task_type == "bin":
        extractor = ClassificationMetaFeatures
        kwargs = bin_kwargs.copy()
    elif args.task_type == "multi":
        extractor = ClassificationMetaFeatures
        kwargs = multi_kwargs.copy()
    elif args.task_type == "regr":
        extractor = RegressionMetaFeatures
        kwargs = regr_kwargs.copy()
    
    dataset_ids = pd.read_csv(Path(__file__).parent / "training" / "dataset_ids" / f"{args.task_type}_dids.csv")["did"].to_list()
    run_batch(
        dataset_ids = dataset_ids,
        task_type = args.task_type,
        extractor = extractor,
        **kwargs
    )






