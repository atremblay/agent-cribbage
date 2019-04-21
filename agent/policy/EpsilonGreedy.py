from .policy import Policy
from .register import register
import numpy as np
import random
import torch


@register
class EpsilonGreedy(Policy):

    def __init__(self, epsilon, decay=1.0):
        self.epsilon = epsilon
        self.decay = decay
        super(EpsilonGreedy).__init__()

    @property
    def custom_hash(self):
        return __name__+str(self.epsilon)

    def step(self):
        self.epsilon = self.decay*self.epsilon

    def prob(self, V_s):
        if self.epsilon <= random.uniform(0, 1):
            return int(np.argmax(V_s))
        else:
            return random.randint(0, len(V_s) - 1)

    def get_transition_probabilities(self, V_s):
        transition_prob = torch.zeros(len(V_s)) + self.epsilon/len(V_s)
        argmax = int(torch.argmax(V_s))
        transition_prob[argmax] += (1 - self.epsilon)
        return transition_prob


