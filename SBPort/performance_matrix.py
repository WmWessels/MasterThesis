from sklearn.model_selection import cross_val_score
from .utils import get_openml_data

NUMBER_CV = 10

# input: list of pipelines
# list of openml datasets
# for every data set, we get the data, and run every pipeline on it
# we return a list of scores for every data set

def run_pipelines(pipelines, datasets, scoring):
    scores = []
    for dataset in datasets:
        X, y, _, _ = get_openml_data(dataset)
        for pipeline in pipelines:
            try:
                score = cross_val_score(pipeline, X, y, cv = NUMBER_CV, scoring = scoring).mean()
            except Exception as e:
                print(f"Error on dataset {dataset}, Error: {e}")
                score = 0
            scores.append(score)

    return scores




