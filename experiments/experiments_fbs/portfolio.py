from typing import Optional, List

import pandas as pd
from sklearn.pipeline import Pipeline

class PortFolioBuilder:

    def __init__(self, size: Optional[int] = 25, strategy: Optional[str] = "greedy"):
        self.size = size
        self.strategy = strategy

    def build_portfolio(self, performance_matrix: pd.DataFrame, cluster_indexes: int) -> List[Pipeline]:
        df_filtered = performance_matrix.loc[cluster_indexes]
        if self.strategy == "greedy":
            average_cluster_performances = df_filtered.mean(axis = 0)
            best_performing = average_cluster_performances.sort_values()[:self.size]
            best_performing_pipelines = list(best_performing.columns)
        
        return best_performing_pipelines