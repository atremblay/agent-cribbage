from .policy import Policy
from .register import register
import numpy as np
import random


@register
class EpsilonGreedy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(EpsilonGreedy).__init__()

    @property
    def custom_hash(self):
        return __name__+str(self.epsilon)

    def prob(self, V_s):
        if self.epsilon <= random.uniform(0, 1):
            return int(np.argmax(V_s))
        else:
            return random.randint(0, len(V_s) - 1)

