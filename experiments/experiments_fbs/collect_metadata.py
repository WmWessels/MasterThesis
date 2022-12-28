import random
import pandas as pd
import openml

class MetaDataSet:

    def __init__(self, openml_indexes: dict):
        self.openml_indexes: dict = openml_indexes
        self.meta_dataset = []
    
    def create_meta_dataset(self, sample_count: int = 300, override: bool = True):
        meta_dataset = []
        for index_ in self.openml_indexes:
            dataset_obj = openml.datasets.get_dataset(index_)
            meta_data = dataset_obj.qualities
            meta_dataset.append(meta_data)
        
        if override:
            self.meta_dataset = meta_dataset
        
        return meta_dataset

def main():
    datasets = openml.datasets.list_datasets()
    sample_count = 50
    dataset_indexes = list(datasets.keys())[0:sample_count]
    # dataset_indexes = random.choices(list(datasets.keys()), k = sample_count)
    MDS = MetaDataSet(dataset_indexes)
    meta_dataset = MDS.create_meta_dataset()
    meta_dataframe = pd.DataFrame(index = dataset_indexes, data = meta_dataset)
    meta_dataframe.to_csv("meta_dataframe_experimental.csv")

if __name__ == "__main__":
    main()