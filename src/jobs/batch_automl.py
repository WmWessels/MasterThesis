import os
import random
import openml
import pandas as pd

from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from gama import GamaClassifier
from gama.gama import Gama

random.seed(0)


class AutomlExecutor:
    
    def run_automl(self, dataset_id: int, max_total_time: Optional[int] = 150, store: Optional[str] = "logs", evaluation_metric: Optional[str] = "accuracy"):
        output_directory = os.path.join(os.getcwd(), f"/logs/gama_{dataset_id}")
        if not os.path.exists("logs"):
            os.mkdir("logs")
        gama_instance = GamaClassifier(max_total_time = max_total_time, store = store, output_directory = output_directory)
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
        
        gama_instance.fit(X_train, y_train)
        if evaluation_metric == "accuracy":

            label_predictions = self.gama_instance.predict(X_test)
            # probability_predictions = self.gama_instance.predict_proba(X_test)
            accuracy = accuracy_score(y_test, label_predictions)
            print(f"""GAMA finished search on dataset {dataset_id} with {max_total_time} seconds time. 
                    The accuracy score is: {accuracy}""")

            return accuracy
        else:
            raise ValueError(f"The given evaluations metric {evaluation_metric} is unknown, please try a valid metric")
    
    def run_batch(self, dataset_ids: List[int], store_to_file: bool = False):
        batch_scores = []
        for dataset_id in dataset_ids:
            try:
                score = self.run_automl(dataset_id)
            except:
                score = 0
            batch_scores.append(score)
        
        if store_to_file:
            pd.DataFrame(index = dataset_ids, data = batch_scores).to_csv("../data/gama_runs.csv")
        return batch_scores

def main():
    dataset_ids = [2, 3]
    automl_executor = AutomlExecutor()
    batch_results = automl_executor.run_batch(dataset_ids)
    print(batch_results)

if __name__== "__main__":
    main()