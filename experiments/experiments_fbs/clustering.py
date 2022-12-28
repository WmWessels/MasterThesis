import pandas as pd
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

def main():
    dataset = pd.read_csv("meta_dataframe_experimental.csv", index_col = 0)
    n_clusters = 3
    clustering_obj = KMeans(n_clusters = n_clusters)
    metadata_cluster = MetaDataCluster(dataset)
    fitted_clustering = metadata_cluster.compute_clusters(clustering_obj)
    cluster_labels = fitted_clustering.labels_
    
if __name__== "__main__":
    main()
    