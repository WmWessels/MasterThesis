import os
import pandas as pd

def main():
    data_directory = "../data"
    log_directory = "../logs"
    subfolders = [f.path for f in os.scandir(log_directory) if f.is_dir()]
    candidate_pipelines = []
    for folder in subfolders:
        current_frame = pd.read_csv(f"{folder}/evaluations.log", delimiter = ';')
        non_failures = current_frame[current_frame["error"] == "None"]
        for pipeline in non_failures["pipeline"]:
            candidate_pipelines.append(pipeline)
    
    pipeline_frame = pd.DataFrame(candidate_pipelines)
    pipeline_frame.to_csv(f"{data_directory}/candidate_pipelines.csv")
    
if __name__ == "__main__":
    main()