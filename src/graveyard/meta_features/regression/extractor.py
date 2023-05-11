import json
from pymfe.mfe import MFE
import openml
import numpy as np
import pandas as pd
from pymfe.info_theory import MFEInfoTheory
from landmarking import LandmarkingRegressor
import scipy
from sklearn.impute import SimpleImputer


class RegressionExtractor:

    def __init__(self, pymfe: MFE, custom_extractor, general_extractor):
        self.pymfe = pymfe

        self.custom_extractor = custom_extractor

        self.general_extractor = general_extractor

        with open("../metafeatures.json", "r") as file:
            file_content = json.load(file)
        self.metafeatures = {k: None for k in file_content["regr"]}

        with open("../func_reference.json", "r") as file:
            function_dict = json.load(file)
        self.regression_functions_general = function_dict["regression"]["general"]
        self.regression_functions_landmarking = function_dict["regression"]["landmarking"]
        self.general_functions = function_dict["always"]

    def retrieve(self, X, y, cat_mask):
        """
        Retrieve metafeatures from the dataset
        """
        names, features = self._run_pymfe(X, y)
        for name, feature in zip(names, features):
            self.metafeatures[name] = feature
        
        self._run_regressor_funcs(X, y, cat_mask)
        self._backfill_missing(X, y)
        self._impute_failures()
        return self.metafeatures

    def _run_pymfe(self, X, y):
        self.pymfe.fit(X, y, precomp_groups = None)
        columns, features = self.pymfe.extract()
        return columns, features
            
    def _run_regressor_funcs(self, X, y, cat_mask):
        X, imputed_x, y = self.impute(X, y, cat_mask)
        for feature, func in self.regression_functions_general.items():
            try:
                self.metafeatures[feature] = getattr(self.custom_extractor, func)(X = X, y = y)
            except:
                self.metafeatures[feature] = np.nan
        for feature, func in self.regression_functions_landmarking.items():
            try:
                self.metafeatures[feature] = getattr(self.custom_extractor, func)(imputed_x, y)
            except:
                self.metafeatures[feature] = np.nan
    
    def impute(self, X: np.array, y: np.array, cat_mask: list[bool]):

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
                self.metafeatures[feature] = getattr(self.general_extractor, func)(X, y)
        
    def _impute_failures(self):
        for feature, value in self.metafeatures.items():
            if value is None:
                self.metafeatures[feature] = "imputed"
