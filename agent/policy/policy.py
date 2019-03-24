import numpy as np


class Policy:
    @staticmethod
    def random_choice(probability):
        return np.random.choice(np.arange(0, len(probability)), p=probability)
