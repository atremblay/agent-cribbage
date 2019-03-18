from .policy import Policy
import random
import numpy as np
from .register import register


@register
class EpsilonGreedy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        super(EpsilonGreedy).__init__()

    def prob(self, Q_s):
        if self.epsilon < random.uniform:
            return self.ramdom_choice(np.array([self.epsilon/len(Q_s)]*len(Q_s))) #Todo: to verify
        else:
            return np.argmax(Q_s)