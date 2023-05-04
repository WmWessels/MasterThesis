import os
import random
import openml
import pandas as pd
import scipy

from typing import List, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, mean_squared_error

from gama import GamaClassifier, GamaRegressor
from gama.gama import Gama
from gama.utilities.metrics import scoring_to_metric

from functools import partial

random.seed(0)
import sklearn


# rmse = scoring_to_metric("neg_root_mean_squared_error")
# print(rmse)

def force_get_dataset(dataset_id=None, *args, **kwargs):
    """ Remove any existing local files about `dataset_id` and then download new copies. """
    did_cache_dir = openml.utils._create_cache_directory_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, dataset_id, )
    openml.utils._remove_cache_dir_for_id(openml.datasets.functions.DATASETS_CACHE_DIR_NAME, did_cache_dir)
    return openml.datasets.get_dataset(dataset_id, *args, **kwargs)

class AutomlExecutor:
    
    def run_automl(self, dataset_id: Union[int, str], max_total_time: Optional[int] = 3600, store: Optional[str] = "logs", evaluation_metric = "neg_log_loss"):
        output_directory = "src/logs/multiclass/gama_" + str(dataset_id) + "/"
        gama_instance = GamaClassifier(max_total_time = max_total_time, store = store, output_directory = output_directory, scoring = evaluation_metric)
        dataset = force_get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
        y = pd.Series(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        
        gama_instance.fit(X_train, y_train)
        if evaluation_metric == "accuracy":

            label_predictions = gama_instance.predict(X_test)
            # probability_predictions = self.gama_instance.predict_proba(X_test)
            accuracy = accuracy_score(y_test, label_predictions)
            print(f"""GAMA finished search on dataset {dataset_id} with {max_total_time} seconds time. 
                    The accuracy score is: {accuracy}""")

            return accuracy
        elif evaluation_metric == "roc_auc":
            probability_predictions = gama_instance.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probability_predictions)
            print(f"""GAMA finished search on dataset {dataset_id} with {max_total_time} seconds time. 
                    The roc_auc score is: {roc_auc}""")

            return roc_auc
        
        elif evaluation_metric == "neg_log_loss":
            probability_predictions = gama_instance.predict_proba(X_test)
            log_loss_score = log_loss(y_test, probability_predictions)
            print(f"""GAMA finished search on dataset {dataset_id} with {max_total_time} seconds time. 
                    The log_loss score is: {log_loss_score}""")

            return log_loss_score
            
        elif evaluation_metric == "neg_root_mean_squared_error":
            predictions = gama_instance.predict(X_test)
            score = mean_squared_error(y_test, predictions, squared = False)
            print(f"""GAMA finished search on dataset {dataset_id} with {max_total_time} seconds time.
                    The rmse score is: {score}""")
    
    def run_batch(self, dataset_ids: List[int], store_to_file: bool = False):
        batch_scores = []
        for dataset_id in dataset_ids:
            try:
                score = self.run_automl(int(dataset_id))
            except Exception as e:
                print(f"Error while running GAMA on dataset {dataset_id}: {e}")
                score = 0
            batch_scores.append(score)
        
        if store_to_file:
            pd.DataFrame(index = dataset_ids, data = batch_scores).to_csv(os.getcwd() + "/src/data/gama_runs.csv")
        return batch_scores

def main():
    binary_ids = pd.read_csv("src/jobs/multiclass_names.csv").iloc[:, 0].values
    automl_executor = AutomlExecutor()
    batch_results = automl_executor.run_batch(binary_ids, store_to_file = True)
    print(batch_results)

if __name__== "__main__":
    main()