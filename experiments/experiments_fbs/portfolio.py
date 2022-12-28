class PortFolioBuilder:

    def __init__(self, size: int = 25, strategy: str = "greedy"):
        self.size = size
        self.strategy = strategy

    def build_portfolio(self, candidate_pipelines, dataset_indexes):
        #for every candidate pipeline
        #average out the performance on the datasets of interest
        #greedily select the best performing
        #add this to the portfolio
        pass
