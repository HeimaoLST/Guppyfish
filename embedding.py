from this import s


class embedding:
    vocab_size: int
    dimensions: int
    weight: list

    def __init__(self, size, dimensions) -> None:
        self.vocab_size = size
        self.dimensions = dimensions
        self.weight = [[]]
