from .policy import Policy
from .register import register
import numpy as np

@register
class Boltzmann(Policy):

    def __init__(self, tao):
        self.tao = tao
        super(Boltzmann).__init__()

    def prob(self, Q_s):
        Q_s_Softmax = self.softmax(np.array(Q_s))
        return int(self.random_choice(Q_s_Softmax))

    def softmax(self, x, axis=None):
        "Stable definition of Softmax"
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x / self.tao)
        return y / y.sum(axis=axis, keepdims=True)

    @property
    def custom_hash(self):
        return __name__+str(self.tao)

