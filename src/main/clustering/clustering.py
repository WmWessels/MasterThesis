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

if __name__ == "__main__":
    meta_data = pd.read_csv("meta_data.csv", index_col = 0)
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.kernel_approximation import RBFSampler
    import matplotlib.pyplot as plt
    scaler = StandardScaler()
    scaled_meta_data = scaler.fit_transform(meta_data)
    # cluster = MetaOPTICS(min_samples = 25, xi = 0.05, min_cluster_size = 0.05)
    cluster = MetaKernelKMeans(n_clusters = 5, kernel = "rbf", gamma = 0.1)
    cluster.fit(scaled_meta_data)
    labels = cluster.labels_

    rbfs = RBFSampler(gamma = 0.1)
    kernelized_data = rbfs.fit_transform(scaled_meta_data)

    tsne = TSNE(n_components = 2)
    reduced_data = tsne.fit_transform(kernelized_data)

    plt.scatter(reduced_data.T[0], reduced_data.T[1], c = labels)
    plt.show()
    print()
    print(np.unique(cluster.labels_, return_counts = True))