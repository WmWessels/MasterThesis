import random
import openml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

from gama import GamaClassifier
from gama.gama import Gama

random.seed(0)


class AutomlExecutor:

    def __init__(self, gama_instance: Gama):

        self.gama_instance = gama_instance
    
    def run_automl(self, index_: int):
        dataset = openml.datasets.get_dataset(index_)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe", target = dataset.default_target_attribute)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

        self.gama_instance.output_directory = f"logs/gama_{index_}"   
        self.gama_instance.fit(X_train, y_train)
        label_predictions = self.gama_instance.predict(X_test)
        probability_predictions = self.gama_instance.predict_proba(X_test)
        print('accuracy:', accuracy_score(y_test, label_predictions))
        print('log loss:', log_loss(y_test, probability_predictions))
        # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
        print('log_loss', self.gama_instance.score(X_test, y_test))
        return accuracy_score(y_test, label_predictions)


def main():
    good_indexes = [44603, 44754, 44478, 44540, 44728, 44558, 44621,   934, 44687,
              913, 44407, 44542, 44426,  4329, 44651,   316,   818, 44450,
            44509, 44611, 44484, 44365, 44587, 44680,  1020, 40670, 44541,
            44544, 44513, 44464]
    max_total_time = 150
    store = 'logs'
    evaluations = []
    for dataset_index in good_indexes:
        gamaclassifier = GamaClassifier(max_total_time = max_total_time, store = store, output_directory = f"../logs/gama_{dataset_index}")
        automl_instance = AutomlExecutor(gamaclassifier)
        try:
            accuracy = automl_instance.run_automl(dataset_index)
        except:
            accuracy = 0
            continue
        print(accuracy)
        evaluations.append(accuracy)
    print(evaluations)
    pd.DataFrame(evaluations).to_csv("gama_runs.csv")
if __name__== "__main__":
    main()