from typing import Optional, List

import pandas as pd
from sklearn.pipeline import Pipeline

class PortFolioBuilder:

    def __init__(self, size: Optional[int] = 25, strategy: Optional[str] = "greedy"):
        self.size = size
        self.strategy = strategy

    def build_portfolio(self, performance_matrix: pd.DataFrame, cluster_indexes: List[int]) -> List[Pipeline]:
        df_filtered = performance_matrix.loc[cluster_indexes]
        if self.strategy == "greedy":
            average_cluster_performances = df_filtered.mean(axis = 0)
            best_performing = average_cluster_performances.sort_values(ascending=True)[:self.size]
            best_performing_pipelines = pd.DataFrame(best_performing)
        
        return best_performing_pipelines
    
def main() -> None:
    data_directory = "../data"
    data = pd.read_csv(f"{data_directory}/batch_results.csv", index_col = 0)
    indexes= pd.read_csv(f"{data_directory}/good_indexes.csv", index_col = 0).iloc[:, 0]
    pipelines = pd.read_csv(f"{data_directory}/pipelines.csv", index_col = 0)
    df = pd.DataFrame(index = indexes, data = data.values, columns = pipelines)
    builder = PortFolioBuilder()
    builder.build_portfolio(df, [2,8])
