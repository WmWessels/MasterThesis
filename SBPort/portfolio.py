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
            best_performing = average_cluster_performances.sort_values(ascending=False)[:self.size]
            best_performing_pipelines = pd.DataFrame(best_performing)
        else:
            raise ValueError(f"""Portfolio building strategy {self.strategy} not recognized, 
                               please try another strategy""")

        return best_performing_pipelines
