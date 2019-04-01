from .policy import Policy
import random
import numpy as np
from .register import register


@register
class EpsilonGreedy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(EpsilonGreedy).__init__()

    @property
    def custom_hash(self):
        return __name__+str(self.epsilon)

    def prob(self, V_s):
        argmax = int(np.argmax(V_s))
        prob = [(self.epsilon)/len(V_s) for _ in range(len(V_s))]
        prob[argmax] += (1.0-self.epsilon)
        if self.epsilon <= random.uniform(0, 1):
            idx = argmax
        else:
            idx = random.randint(0, len(V_s) - 1)

        return idx, prob

