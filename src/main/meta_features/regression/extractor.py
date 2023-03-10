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
        self.metafeatures = {k: None for k in file_content["regression"]}

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


# if __name__ == "__main__":

    
#     mfe = MFE(features = features, groups = ["general", "statistical", "info-theory"], summary = ["nanmean", "nansd"])
    
#     regression_sets = openml.tasks.list_tasks(task_type= openml.tasks.TaskType.SUPERVISED_REGRESSION, output_format="dataframe")
#     regression_sets["size"] = regression_sets["NumberOfInstances"] * regression_sets["NumberOfFeatures"]
#     small_indexes = set(regression_sets.query("NumberOfInstances < 20").groupby('did').first().index)
#     large_indexes = set(regression_sets.query("NumberOfInstances > 500_000 or NumberOfFeatures > 2500 or size > 10_000_000").groupby('did').first().index)
#     indexes_to_filter = small_indexes.union(large_indexes)
#     indexes = list(map(int, set(regression_sets['did'].unique()).difference(indexes_to_filter)))
#     counter = 0
#     def batch(iterable , batch_size: int):
#         for i in range(0, len(iterable), batch_size):
#             yield iterable[i: i + batch_size]
#     for index_ in batch(indexes, 20):

#         meta_features = []
#         for ind in index_:
#             try:
#                 regr = RegressionMetaFeatures(LandmarkingRegressor())
#                 gen = GenericMetaFeatures(MFEInfoTheory())

#                 extractor = RegressionExtractor(mfe, regr, gen)
#                 dataset = openml.datasets.get_dataset(ind)
#                 X, y, cat_mask, _ = dataset.get_data(dataset_format="array", target = dataset.default_target_attribute)
#                 if type(X) == scipy.sparse._csr.csr_matrix:
#                     X = X.toarray()
#                 mf = extractor.retrieve(X, y, cat_mask)
#                 meta_features.append(mf)
#             except Exception as e:
#                 print(e)
#                 meta_features.append({k: np.nan for k in extractor.metafeatures.keys()})
    
#         pd.DataFrame(index = index_, data = meta_features).to_csv(f"../data/regression/meta_features_{counter}.csv")
#         counter += 1
    