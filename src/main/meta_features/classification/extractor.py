"""Workflow for calculating meta features, including recomputing and imputing to ensure a high quality meta feature vector"""
import numpy as np
import pandas as pd

import json

from sklearn.impute import SimpleImputer

from typing import List

from pymfe.mfe import MFE

class ClassificationExtractor:

    def __init__(self,
                 pymfe: MFE,
                 pymfe_lm: MFE,
                 custom_extractor,
                 general_extractor,
                 ):
        self.pymfe = pymfe
        self.pymfe_lm = pymfe_lm
        self.custom_extractor = custom_extractor
        self.general_extractor = general_extractor

        with open("utils/metafeatures.json", "r") as file:
            file_content = json.load(file)
        self.metafeatures = {k: None for k in file_content["clf"]}

        with open("utils/func_reference.json", "r") as file:
            function_dict = json.load(file)
        self.classification_functions_general = function_dict["classification"]["general"]
        self.general_functions = function_dict["always"]
    
    def retrieve(self, X, y, cat_mask):
        """
        Retrieve metafeatures from the dataset
        """
        X, imputed_X, y = self.impute(X, y, cat_mask)

        names, features = self._run_pymfe(X, y, imputed_X)
        for name, feature in zip(names, features):
            self.metafeatures[name] = feature
        
        self._run_classification_funcs(X, y)
        self._backfill_missing(X, y)
        self._impute_failures()
        return self.metafeatures

    def _run_pymfe(self, X: np.array, y: np.array, imputed_X: np.array):
        """calculate metafeatures using pymfe library"""

        # calculate meta features

        self.pymfe.fit(X, y, precomp_groups = None)
        self.pymfe_lm.fit(imputed_X, y, precomp_groups=None)

        columns, features = self.pymfe.extract()
        columns_lm, features_lm = self.pymfe_lm.extract()
        
        columns.extend(columns_lm)
        features.extend(features_lm)

        return columns, features

    def _run_classification_funcs(self, X: np.array, y: np.array):

        for feature, func in self.classification_functions_general.items():
            try:
                self.metafeatures[feature] = getattr(self.custom_extractor, func)(X = X, y = y)
            except:
                self.metafeatures[feature] = np.nan
            
    @staticmethod 
    def impute(X: np.array, y: np.array, cat_mask: List[bool]):

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

    def _backfill_missing(self, X, y):
        for feature, func in self.general_functions.items():
            if pd.isna(self.metafeatures[feature]):
                self.metafeatures[feature] = getattr(self.general_extractor, func)(X = X, y = y)
        
    def _impute_failures(self):
        for feature, value in self.metafeatures.items():
            if pd.isna(value):
                self.metafeatures[feature] = "imputed"
    
