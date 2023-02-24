import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.cluster import KMeans

class MetaDataCluster:

    def __init__(self, dataset: DataFrame):
        self.dataset = dataset



    def save_to_file(self):
        """
        Future function to save clustering model to a file as pmml
        """
        pass
