import random
import pandas as pd
import openml

random.seed(0)

class MetaDataSet:

    def __init__(self, openml_indexes: dict):
        self.openml_indexes: dict = openml_indexes
        self.meta_dataset = []
    
    def create_meta_dataset(self, override: bool = True):
        meta_dataset = []
        for index_ in self.openml_indexes:
            print(index_)
            dataset_obj = openml.datasets.get_dataset(index_)
            meta_data = dataset_obj.qualities
            meta_dataset.append(meta_data)
        
        if override:
            self.meta_dataset = meta_dataset
        
        return meta_dataset

def main():
    data_dir = "../data"
    good_indexes = pd.read_csv(f"{data_dir}/good_indexes.csv", index_col = 0).iloc[:, 0]
    # sample_count = 50
    # dataset_indexes = random.sample(good_indexes, sample_count)
    MDS = MetaDataSet(good_indexes)
    meta_dataset = MDS.create_meta_dataset()
    meta_dataframe = pd.DataFrame(index = good_indexes, data = meta_dataset)
    meta_dataframe.to_csv("../data/meta_dataframe_experimental.csv")

if __name__ == "__main__":
    main()