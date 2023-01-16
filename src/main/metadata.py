import os
import random
import pandas as pd
import openml

from typing import List

class MetaDataSet:
    
    @staticmethod
    def create_meta_dataset(dataset_ids: List[int]):
        meta_dataset = []

        for dataset_id in dataset_ids:
            dataset_obj = openml.datasets.get_dataset(dataset_id)
            meta_data = dataset_obj.qualities
            meta_dataset.append(meta_data)

        return meta_dataset