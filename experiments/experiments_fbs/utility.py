from typing import Iterable

def batch(iterable: Iterable , batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i: i + batch_size]
    