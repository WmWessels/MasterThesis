# MasterThesis

This is the main repository for my master thesis. Both the code for my methodology and the experiments are available here.
The code is structured in the src folder as follows:

main folder:
- `clustering.py` Is a module that takes a data set, and outputs a clustering model;
- `metadata.py` Is a module that requests data sets from the OpenML API and stores the meta data;
- `portfolio.py` Takes a cluster of the performance matrix and outputs a portfolio;
- `utils.py` Some small utility functions, one for batching and one timeout wrapper;

jobs:
- `batch_automl.py` Runs GAMA on a set of OpenML data sets;
- `batch_pipeline_evaluation.py` Runs a batch of indexes and candidate pipelines. Saves the performance per batch to avoid large data loss in case of crashes;

experiment:
- `portfolio_experiment.ipynb` Contains the code for running the feasibility study experiments;
- This folder will be extended to a more convenient experimental suite;

data_processing:
- `pipeline_retrieval.py` Parses a GAMA log file and stores all the evaluated pipelines that did not throw an error;

data:
- Saves the pipelines and selected data set ids for now;


