"""Workflow for calculating meta features, including recomputing and imputing to ensure a high quality meta feature vector"""
import numpy as np
import pandas as pd

import time
import openml

from sklearn.impute import SimpleImputer

from typing import List, Optional

from pymfe.mfe import MFE
from metafiller import BackFiller
from metaimputer import MetaImputer

class MetaComputer:

    def __init__(self, mfe: MFE, mfe_lm: MFE, backfiller: BackFiller, imputer: MetaImputer):
        self.mfe = mfe
        self.mfe_lm = mfe_lm
        self.backfiller = backfiller
        self.imputer = imputer

    @classmethod
    def _calc_pymfe(cls, X: np.array, imputed_X: np.array, y: np.array, cat_cols: List[str]):
        """calculate metafeatures using pymfe library"""
        #return calculated metafeatures with its columns, also optionally return a missing column mask
        pass

    @classmethod
    def _backfill_metafeatures(cls, X: np.array, y: np.array, cat_cols: List[str], missing_mask: List[bool] | None = None):
        """backfill metafeatures, typically called after execution of function `calc_metafeatures`, but can also be ran as a replacement for calc_metafeatures"""
        #return backfilled meta features, optionally return column mask with missing values
        pass

    @classmethod
    def _impute_metafeatures(cls, X: np.array, y: np.array, missing_mask: List[bool] | None = None):
        """impute meta features, it's recommended not to run this prior to `calc_metafeatures` or `backfill_metafeatures`, as the quality of the meta feature vector will likely decrease"""
        #return a meta feature vector without nans, need to force this or return an error. cannot return a vector with missing values
        pass

    @classmethod
    def get_metafeatures(cls, X: np.array, imputed_X: np.array, y: np.array, cat_cols: List[str]):

        meta_features, missing_mask = cls._calc_pymfe(X, imputed_X, y, cat_cols)
        meta_features, missing_mask = cls._backfill_metafeatures(X, y, cat_cols, missing_mask )
        meta_features = cls._impute_metafeatures(X, y, missing_mask)
        
        return meta_features

    
    def get_metafeatures_job(cls, indexes: List[int], columns: List[str], save: Optional[bool] = True):
                
        meta_features = []
        for idx in indexes:
            print(f"Starting run for dataset with index: {idx} ...")
            dataset = openml.datasets.get_dataset(idx)
            X, y, cat_mask, attributes = dataset.get_data(dataset_format="array", target = dataset.default_target_attribute)
            cat_cols = [b for a,b in zip(cat_mask, attributes) if a]
            begin_time = time.time()
            X, imputed_X, y = cls._impute(X, y, cat_mask)
            end_time = time.time()
            print(f"finished preprocessing in {end_time - begin_time}")
            begin_calculations = time.time()
            meta_data = cls.get_metafeatures(X, imputed_X, y, cat_cols)
            end_calculations = time.time()
            print(f"finished calculating meta data in {begin_calculations - end_calculations}")
            meta_features.append(meta_data)
        
        cls.meta_features = meta_features

        if save:
            pd.DataFrame(index = list(indexes), data = meta_features, columns = columns).to_csv("metafeatures.csv")

            
    @classmethod    
    def _impute(cls, X: np.array, y: np.array, cat_mask = List[bool]):

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