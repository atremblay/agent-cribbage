import numpy as np


class Policy:
    @staticmethod
    def ramdom_choice(probability):
        return np.random.choice(np.arange(0, len(probability)), p=probability)
