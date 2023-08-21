import shutil
from pathlib import Path
import argparse

import pandas as pd
from sklearn.impute import SimpleImputer

from gama import GamaClassifier
from gama.genetic_programming.components import Individual

from candidate_generation import Task


def convert_str_to_pipeline(pipeline_str: str, clf: GamaClassifier):
    
    individual = Individual.from_string(
        pipeline_str,
        clf._pset,
        clf._operator_set._compile
    )

    pipeline = individual.pipeline
    pipeline.steps.insert(0, ('imputation', SimpleImputer(strategy='median')))
    pipeline_as_str = f"{pipeline}".replace(" ", "").replace("\n", "")

    return pipeline_as_str

def format_score(x):
    try:
        return eval(x)[0]
    except:
        return None

def select_candidate_pipelines(task: Task, clf: GamaClassifier):
    log_path = Path(__file__).parent / "logs" /  task 
    candidates = set()

    for folder in log_path.iterdir():

        #get the integer from the folder name at the end
        dataset_id = str(folder).split("_")[-1]
        current_log = pd.read_csv(f"{folder}/evaluations.log", delimiter = ';')
        current_log["pipeline"] = current_log["pipeline"].apply(lambda x: convert_str_to_pipeline(x, clf))
        current_log.drop_duplicates(subset = "pipeline", inplace = True)
        current_log["score"] = current_log["score"].apply(lambda x: format_score(x)).astype(float)
        
        """
        We ideally want to select the top 5 pipelines directly, but GAMA sometimes evaluates duplicate pipelines.
        Additionally, some pipelines might be identical across different data sets.
        """

        top_ten = current_log.nlargest(10, "score", keep = "first")["pipeline"].to_list()
        counter = 0
        for item in top_ten:
            if item not in candidates:
                candidates.add(item)
                counter += 1
            if counter == 5:
                break
            if item == top_ten[-1]:
                print(f"Could not find 5 unique pipelines for dataset {dataset_id}")
    
    return candidates

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type = str, choices = ["bin", "multi", "regr"], required = True)
    args = parser.parse_args()
    clf = GamaClassifier(max_total_time=3, output_directory = ".temp", store = 'nothing')
    candidates = select_candidate_pipelines(args.task, clf)
    pd.DataFrame(candidates, columns = ["pipelines"]).to_csv(f"candidate_pipelines/candidate_pipelines_{args.task}.csv", index = False)
    shutil.rmtree(".temp")

