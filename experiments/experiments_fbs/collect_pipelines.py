import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Iterator, Literal, Optional

class PipelineReader:

    def __init__(self, log_directory):
        self.log_directory = log_directory

    @staticmethod
    def get_folders(log_directory: str) -> Iterator[str]:
        subfolders = [f.path for f in os.scandir(log_directory) if f.is_dir()]
        return subfolders
    
    def retrieve_pipelines(self, save: Optional[bool] = True, data_directory: Optional[str] = "../data") -> pd.DataFrame:
        candidate_pipelines = []
        for folder in self.get_folders(self.log_directory):
            print(folder)
            current_frame = pd.read_csv(f"{folder}/evaluations.log", delimiter = ';')
            non_failures = current_frame[current_frame["error"].notnull()]
            for pipeline in non_failures["pipeline"]:
                candidate_pipelines.append(pipeline.replace(' ', ''))  
        pipeline_frame = pd.DataFrame(candidate_pipelines)
        if save:
            pipeline_frame.to_csv(f"{data_directory}/candidate_pipelines.csv")
        return pipeline_frame
    
def main():
    data_directory = "../data"
    log_directory = "../logs"
    reader = PipelineReader(log_directory)
    print(list(reader.get_folders(reader.log_directory)))
    candidate_pipelines = reader.retrieve_pipelines()
    
if __name__ == "__main__":
    main()