# import classifierextractor
# import regressionextractor
from typing import Protocol
import openml
import pandas as pd
import numpy as np
import json
from classification.extractor import ClassificationExtractor
from pymfe.mfe import MFE
from pymfe.info_theory import MFEInfoTheory
from metafiller import ClassificationMetaFeatures, GenericMetaFeatures
import scipy
from enum import Enum

class Extractor(Protocol):

    def retrieve() -> dict[str, float]:
        """
        Retrieve metafeatures from the dataset
        """
        ...

class Task(Enum):
    CLASSIFICATION = "clf"
    REGRESSION = "regr"

def batch(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]
    

def main() -> None:

    task = Task.CLASSIFICATION.value
    with open("utils/metafeatures.json", "r") as file:
        file_content = json.load(file)
    features = file_content[f"pymfe_{task}"]
    frame_columns = file_content[f"{task}"]
    indexes = list(map(int, pd.read_csv(f"utils/{task}_indexes.csv", index_col = 0).values[:, 0]))
    if task == "clf":
        features_lm = ["best_node", "linear_discr", "naive_bayes", "random_node", "worst_node"]
    counter = 0
 
    for index_batch in batch(indexes, 20):
        meta_features = []
        for ind in index_batch:
            try:
                dataset = openml.datasets.get_dataset(ind, download_qualities = False)
                X, y, cat_mask, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format='array')
                if type(X) == scipy.sparse._csr.csr_matrix:
                    X = X.to
                mfe = MFE(features = features, summary = ["nanmean", "nansd"])
                mfe_lm = MFE(features = features_lm, groups = ["landmarking"], summary = ["nanmean", "nansd"], num_cv_folds = 5, lm_sample_frac = 0.5, suppress_warnings=True)
                mfe_info = MFEInfoTheory()
                custom_extractor = ClassificationMetaFeatures(mfe_info)
                general_extractor = GenericMetaFeatures(mfe_info)
                extractor = ClassificationExtractor(mfe, mfe_lm, custom_extractor, general_extractor)
                mf = extractor.retrieve(X, y, cat_mask)
                meta_features.append(mf)
                print(mf)
            except Exception as e:
                print(e)
                meta_features.append({col: np.nan for col in frame_columns})

        pd.DataFrame(index = index_batch, data = meta_features).to_csv(f"data/classification/meta_features_{counter}.csv")
        counter += 1

if __name__ == "__main__":
    main()