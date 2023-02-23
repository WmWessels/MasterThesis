import numpy as np
import pandas as pd

import openml

from pymfe.mfe import MFE
from pymfe.info_theory import MFEInfoTheory

from meta_features.metafiller import BackFiller
from metafeature_mapping import mapping_general, mapping_stat, mapping_infotheory, mapping_landmarking

mapping = mapping_general | mapping_stat | mapping_infotheory | mapping_landmarking

class MetaImputer:

    def __init__(self, mfe_info, backfiller: BackFiller):
        self.mfe_info = mfe_info
        self.backfiller = backfiller
    
    def impute_infotheory():
        pass

    def impute_landmarking():
        pass
    
    @classmethod
    def impute_general_stat(cls, feature: str, X: np.array):
        func_str = mapping[feature]
        func = getattr(cls.backfiller, func_str)
        result = func(X)
        return result

    def get_missing(df: pd.DataFrame):
        missing_columns = np.unique(list(zip(*df.loc[:, df.isnull().any()].columns.str.split('.'))))
        return missing_columns
    
    @classmethod
    def impute_metafeatures(cls, X, y, df, missing_columns):
        imputed_values = []
        for feature in missing_columns:
            if feature in mapping_infotheory.keys():
                result = cls.impute_infotheory()
                "set df column"
            elif feature in mapping_landmarking.keys():
                result = cls.impute_landmarking()
                "set df column"
            else:
                result = cls.impute_general_stat(feature, X)
            
            func = mapping[feature]




    
def impute_metafeatures(X, y, df):
    missing_elements = np.unique(np.array(list(zip(*df.loc[:, df.isnull().any()].columns.str.split('.')))))
    for element in missing_elements:
        if element in mapping_infotheory.keys():
            imputing_class = mfe_info
            value = mapping[element]
            func, args = value[0], value[1:]
            C = pd.DataFrame(X).select_dtypes('category')
            C = C.to_numpy()
            if len(args) > 1:
                args = C, y
            else:
                args = C
        else:
            imputing_class = filler
            func = mapping[element]
            args = X
        if not element in mapping_landmarking.keys():
            exec_func = getattr(imputing_class, func)
            result = exec_func(args)
        else:
            result = mapping[element]
        if type(result) == tuple:
            column_str = [f"{element}.nanmean", f"{element}.nansd"]
        else:
            df[element] = result
            continue
        for col, res in zip(column_str, result):
            df[col] = res
    return df




def main() -> None:

    meta_data = pd.read_csv("meta_features_total.csv", index_col = 0)
    datasets = openml.datasets

    mfe_info = MFEInfoTheory()
    filler = BackFiller()

    imputer = MetaImputer(mfe_info, filler)

    for inx in meta_data.index:

        dataset = datasets.get_dataset(inx, download_all_files=False)
        X, y, _, _ = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)
        df = pd.DataFrame(columns = meta_data.columns)
        df.loc[inx] = meta_data.loc[inx].values

        row_ = impute_metafeatures(X, y, df)
        print(row_)
        meta_data.loc[inx] = row_.values

if __name__ == "__main__":
    main()