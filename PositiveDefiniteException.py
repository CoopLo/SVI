class PositiveDefiniteException(Exception):

    def __init__(self, iteration, symmetric, positive_definite):
        self.iteration = iteration
        self.symmetric = symmetric
        self.positive_definite = positive_definite

