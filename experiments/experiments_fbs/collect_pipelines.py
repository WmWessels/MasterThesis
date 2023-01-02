import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from gama.logging.GamaReport import *

def main():
    data_directory = "../data"
    log_directory = "../logs"
    subfolders = [f.path for f in os.scandir(log_directory) if f.is_dir()]
    candidate_pipelines = []
    # gama_report = GamaReport("/home/wichert/Master DS&AI/master_thesis/experiments/logs/gama_2")
    # gama_report.update(force=True)
    # print(gama_report.successful_evaluations)
    # print(gama_report.successful_evaluations["pipeline"])
    for folder in subfolders:
        # gama_report = GamaReport(folder)
        # gama_report.update()
        # print(gama_report.evaluations)
        current_frame = pd.read_csv(f"{folder}/evaluations.log", delimiter = ';')
        non_failures = current_frame[current_frame["error"].notnull()]
        for pipeline in non_failures["pipeline"]:
            candidate_pipelines.append(pipeline.replace(' ', ''))   
            # print(candidate_pipelines)
    
    pipeline_frame = pd.DataFrame(candidate_pipelines)
    pipeline_frame.to_csv(f"{data_directory}/candidate_pipelines.csv")
    
if __name__ == "__main__":
    main()