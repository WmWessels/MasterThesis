import numpy as np
import pandas as pd

from typing import Protocol

from pandas import DataFrame
from kernel_kmeans import KernelKMeans

from sklearn.cluster import OPTICS

class MetaCluster(Protocol):

    def load_from_file(self):
        ...

    def save_to_file(self):
        ...
    
    def pred_to_portfolio(self):
        ...

class MetaKernelKMeans(KernelKMeans):
    """ Small wrapper around KernelKMeans to make it more usable for our purposes """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_from_file(self):
        """
        Future function to load clustering model from a file
        """
        pass

    def save_to_file(self):
        """
        Future function to save clustering model to a file
        """
        pass

    def pred_to_portfolio(self):
        """
        Future function to predict the portfolio based on the clustering model
        """
        pass

class MetaOPTICS(OPTICS):
    """ Small wrapper around OPTICS to make it more usable for our purposes """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_from_file(self):
        """
        Future function to load clustering model from a file
        """
        pass

    def save_to_file(self):
        """
        Future function to save clustering model to a file
        """
        pass

    def pred_to_portfolio(self):
        """
        Future function to predict the portfolio based on the clustering model
        """
        pass
