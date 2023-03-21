import numpy as np
import pandas as pd

from enum import Enum

from typing import Optional, Protocol

from pandas import DataFrame
from kernel_kmeans import KernelKMeans

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import OPTICS
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

import skops.io as sio

import matplotlib.pyplot as plt


class Reduction(Enum):
    PCA = "pca"
    TSNE = "tsne"

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
        
    def load_from_file(self, file: str):
        """
        Future function to load clustering model from a file
        """
        obj = sio.load(file, trusted = False)
        if isinstance(obj, MetaKernelKMeans):
            return obj
        else:
            raise TypeError(
                """
                The output of sio.load is not a MetaKernelKMeans object, 
                please assure that the file is safe
                """
            )

    def save_to_file(self, file: Optional[str] = "kernel_kmeans.skops"):
        """
        Future function to save clustering model to a file
        """
        sio.dump(self, file)

    def plot(self, X, reduction_method: Reduction):
        if self.kernel == "rbf":
            X = RBFSampler(gamma = self.gamma, random_state = self.random_state).fit_transform(X)

        if reduction_method == "pca":
            reduced_X = PCA(n_components = 2, random_state = self.random_state).fit_transform(X)
        
        elif reduction_method == "tsne":
            reduced_X = TSNE(n_components = 2, random_state = self.random_state).fit_transform(X)

        plt.scatter(reduced_X.T[0], reduced_X.T[1], c = self.labels_)
        plt.show()


class MetaOPTICS(OPTICS):
    """ Small wrapper around OPTICS to make it more usable for our purposes """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_from_file(self, file: str):
        """
        Future function to load clustering model from a file
        """
        obj = sio.load(file, trusted = False)
        if isinstance(obj, MetaOPTICS):
            return obj
        else:
            raise TypeError(
                """
                The output of sio.load is not a MetaOPTICS object, 
                please assure that the file is safe
                """
            )

    def save_to_file(self, file: Optional[str] = "optics.skops"):
        """
        Future function to save clustering model to a file
        """
        if not type(file) == str:
            raise TypeError("The filename must be a string")
        
        sio.dump(self, file)


if __name__ == "__main__":
    """
    Example of how to use the clustering models work and how to visualize the clusters
    """
    meta_data = pd.read_csv("meta_data.csv", index_col = 0).sample(700, random_state = 0)

    meta_kernelkmeans = MetaKernelKMeans(n_clusters = 5, kernel = "rbf", gamma = 0.2, random_state = 0)
    scaled_metadata = MinMaxScaler().fit_transform(meta_data)
    meta_kernelkmeans.fit(scaled_metadata)
    meta_kernelkmeans.save_to_file()

    labels = meta_kernelkmeans.labels_
    print(np.unique(labels, return_counts = True))
    meta_kernelkmeans.plot(scaled_metadata, reduction_method = "tsne")
