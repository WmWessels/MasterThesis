# import classifierextractor
# import regressionextractor
from typing import Protocol

class Extractor(Protocol):

    def retrieve() -> dict[str, float]:
        """
        Retrieve metafeatures from the dataset
        """
        ...

def main(extractor: Extractor) -> None:
    metafeatures = extractor.retrieve()