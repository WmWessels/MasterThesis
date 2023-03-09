"""Workflow for calculating meta features, including recomputing and imputing to ensure a high quality meta feature vector"""
import numpy as np
import pandas as pd

import time
import openml

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from typing import List, Optional, Dict

from pymfe.mfe import MFE
from src.main.meta_features.regression.metafiller import BackFiller

from main.meta_features.func_mapping_clf import *
import os

from src.main.utils import ImputationError

class MetaComputer:

    def __init__(self, 
                 mfe: MFE,
                 mfe_lm: MFE, 
                 backfiller: BackFiller, 
                 scaler: StandardScaler
        ):
        self.mfe = mfe
        self.mfe_lm = mfe_lm
        self.backfiller = backfiller
        self.scaler = scaler

    def _calc_pymfe(self, X: np.array, imputed_X: np.array, y: np.array, cat_cols: list[str]):
        """calculate metafeatures using pymfe library"""

        # calculate meta features
        self.mfe.fit(X, y, cat_cols = cat_cols)
        self.mfe_lm.fit(imputed_X, y, cat_cols = cat_cols)
        columns, features = self.mfe.extract()
        columns_lm, features_lm = self.mfe_lm.extract()

        # extract columns where meta features are missing
        
        columns.extend(columns_lm)
        # mean, sd = self.backfiller.get_missing(X)
        # columns.extend(["missing.nanmean", "missing.nansd"])
        features.extend(features_lm)
        # features.extend([mean, sd])
        missing_columns = [column for column, meta_feature in zip(columns, features) if np.isnan(meta_feature)]
        meta_features = {k: v for k,v in zip(columns, features)}
        return meta_features, missing_columns

    def _backfill_metafeatures(self, 
                               X: np.array, 
                               y: np.array, 
                               cat_cols: list[str], 
                               meta_features: dict[str, float], 
                               missing_columns: list[str] = None
        ) -> dict[str, float]:
        """backfill metafeatures, typically called after execution of function `calc_metafeatures`, but can also be ran as a replacement for calc_metafeatures"""

        for column in missing_columns:
            if not column in mapping_landmarking.keys():
                args = X if column not in mapping_infotheory.keys() else (X[cat_cols], y)           
                func = mapping[column]
                exec_func = getattr(self.backfiller, func)
                result = exec_func(args)
            else:
                result = mapping[column]
            
            meta_features[column] = result
        print(f"recalculated {len(missing_columns)} meta features")
        missing_columns = [column for column, value in meta_features.items() if np.isnan(value)]
        
        return meta_features, missing_columns


    @classmethod
    def _impute_metafeatures(cls, X: np.array, y: np.array, meta_features: dict[str, float], missing_columns: List[str] = None):
        """impute meta features, it's recommended not to run this prior to `calc_metafeatures` or `backfill_metafeatures`, as the quality of the meta feature vector will likely decrease"""
        for column in missing_columns:
            meta_features[column] = "impute value"
        

    def get_metafeatures(self, X: np.array, imputed_X: np.array, y: np.array, cat_cols: List[str]):

        meta_features, missing_columns = self._calc_pymfe(X, imputed_X, y, cat_cols)
        if any(missing_columns):
            meta_features, missing_columns = self._backfill_metafeatures(X, y, cat_cols, meta_features, missing_columns)
        if any(missing_columns):
            meta_features = self._impute_metafeatures(X, y, missing_columns)

        scaled_meta_features = self.scaler.transform([list(meta_features.values())])

        return meta_features.keys(), scaled_meta_features[0]

    
    def get_metafeatures_job(cls, indexes: List[int], save: Optional[bool] = True):
                
        meta_features = []
        for idx in indexes:
            print(f"Starting run for dataset with index: {idx} ...")
            dataset = openml.datasets.get_dataset(idx)
            X, y, cat_mask, attributes = dataset.get_data(dataset_format="array", target = dataset.default_target_attribute)
            cat_cols = [b for a,b in zip(cat_mask, attributes) if a]
            begin_time = time.time()
            X, imputed_X, y = cls.impute(X, y, cat_mask)
            end_time = time.time()
            print(f"finished preprocessing in {end_time - begin_time}")
            begin_calculations = time.time()
            columns, meta_data = cls.get_metafeatures(X, imputed_X, y, cat_cols)
            end_calculations = time.time()
            print(f"finished calculating meta data in {end_calculations - begin_calculations}")
            meta_features.append(meta_data)
            print(meta_data)
        cls.meta_features = meta_features

        if save:
            pd.DataFrame(index = indexes, data = meta_features, columns = columns).to_csv("metafeatures.csv")

            
    @classmethod    
    def impute(cls, X: np.array, y: np.array, cat_mask: List[bool]):

        num_mask = [not elem for elem in cat_mask]

        C = X[:, cat_mask]
        N = X[:, num_mask]

        imputer_cat = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        if len(C[0, :]) > 0:
            C = imputer_cat.fit_transform(C)

        imputer_num = SimpleImputer(strategy='mean', missing_values=np.nan)
        if len(N[0, :]) > 0:
            N = imputer_num.fit_transform(N)

        imputed_X = np.concatenate((C, N), axis = 1)
        
        if y is None:
            y = X[:, -1]
            imputed_X = imputed_X[:, :-1]
            X = X[:, :-1]

        return X, imputed_X, y
    

if __name__ == "__main__":
    indexes = [2, 3]
    features = ["num_to_cat", "nr_class", "freq_class", "nr_attr", "nr_bin", "nr_cat", "nr_inst", \
                "nr_num", "cor", "cov", "iq_range", "kurtosis", "max", "mean", "median", "min",   \
                "nr_outliers", "sd", "skewness", "var", "attr_ent", "joint_ent", "eq_num_attr", "class_conc", "attr_conc"]
    features_lm = ["best_node", "linear_discr", "naive_bayes", "random_node", "worst_node"]

    mfe = MFE(features = features, summary = ["nanmean", "nansd"])
    mfe_lm = MFE(features = features_lm, groups = ["landmarking"], summary = ["mean", "sd"], num_cv_folds = 5, lm_sample_frac = 0.5, suppress_warnings=True)
    backfiller = BackFiller()
    from sklearn.preprocessing import MinMaxScaler
    meta_data = pd.read_csv("meta_data.csv", index_col = 0)
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(meta_data)
    mf_calculator = MetaComputer(mfe = mfe, mfe_lm = mfe_lm, backfiller = backfiller, scaler = scaler)
    mf_calculator.get_metafeatures_job(indexes = indexes)