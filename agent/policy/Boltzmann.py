from .policy import Policy
import numpy as np
from .register import register


@register
class Boltzmann(Policy):

    def __init__(self, tao):
        self.tao = tao
        super(Boltzmann).__init__()

    def prob(self, Q_s):
        Q_s_Softmax = self.softmax(Q_s, self.tao)
        return self.ramdom_choice(Q_s_Softmax), Q_s_Softmax

    def softmax(self, x, axis=None):
        "Stable definition of Softmax"
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x / self.tao)
        return y / y.sum(axis=axis, keepdims=True)

    @property
    def custom_hash(self):
        return __name__+str(self.tao)

