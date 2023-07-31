# gama with portfolio warm start vs vanilla gama.
from gama import GamaClassifier
import logging
import sys
from utils import prepare_openml_for_inf

dataset_id = 3 
X, y, _ = prepare_openml_for_inf(dataset_id)
max_total_time = 120 
store = "logs"
evaluation_metric = "roc_auc"

clf = GamaClassifier(max_total_time = max_total_time, store = store, scoring = evaluation_metric)
clf.fit(X, y)

