import numpy as np
import pandas as pd
import pickle

from pandas import DataFrame
from sklearn.cluster import KMeans

class MetaDataCluster:

    def __init__(self, dataset: DataFrame):
        self.dataset = dataset
    
    def impute_dataframe(self):
        return self.dataset.apply(lambda x: x.fillna(x.mean(), axis = 0))

    def compute_clusters(self, cluster):
        clean_dataframe = self.impute_dataframe()
        cluster.fit(clean_dataframe)
        return cluster

    def save_to_file(self):
        """
        Future function to save clustering model to a file
        """
        pass
