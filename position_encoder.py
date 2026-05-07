import numpy as np
from numpy.typing import NDArray


class PositionEncoder:
    max_seq: int
    pe: NDArray
    def __init__(self, max_seq, embedding_dim) -> None:

    def _build_pe(self, max_seq, embedding_dim):
        
