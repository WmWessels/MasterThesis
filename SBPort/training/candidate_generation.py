import argparse
from typing import Optional, Union
from enum import Enum
from pathlib import Path
import random

import openml
import pandas as pd

from gama import GamaClassifier, GamaRegressor
from gama.gama import Gama

class Metric(Enum):
    ROC_AUC = "roc_auc"
    LOG_LOSS = "neg_log_loss"
    RMSE = "neg_root_mean_squared_error"

class Task(Enum):
    BINARY = "bin"
    MULTICLASS = "multi"
    REGRESSION = "regr"

random.seed(42)

class AutomlExecutor:

    def __init__(self, task: Task):
        self._task = task
    
    def _run_automl(self, dataset_id: Union[int, str], max_total_time: Optional[int] = 3600, store: Optional[str] = "logs", evaluation_metric: Metric = Metric.ROC_AUC):
        output_directory = Path(__file__).parent / "logs" / self._task / f"gama_{dataset_id}"
        if not output_directory.exists():
            output_directory.mkdir(parents = True)
        if self._task == Task.REGRESSION:
            gama_instance = GamaRegressor(max_total_time = max_total_time, store = store, output_directory = output_directory, scoring = evaluation_metric)
        else:
            gama_instance = GamaClassifier(max_total_time = max_total_time, store = store, output_directory = output_directory, scoring = evaluation_metric)
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
        if "sparse" in dataset.format.lower():
            X = X.sparse.to_dense()
        y = pd.Series(y)
        gama_instance.fit(X, y)
 
    def run_batch(self, dataset_ids: list[Union[int, str]]):

        for dataset_id in dataset_ids:
            try:
                self._run_automl(dataset_id)
            except Exception as e:
                print(f"Error while running GAMA on dataset {dataset_id}: {e}")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = argparser.parse_args()
    dataset_ids = pd.read_csv(Path(__file__).parent / "dataset_ids" / f"{args.task}_dids.csv")["did"].to_list()
    dataset_ids = random.choice(dataset_ids, 50, replace = False)
    automl_executor = AutomlExecutor(args.task)
    automl_executor.run_batch(dataset_ids)

if __name__== "__main__":
    main()