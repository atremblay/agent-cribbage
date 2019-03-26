from .policy import Policy
import random
from .register import register


@register
class Random(Policy):

    def __init__(self):
        super().__init__()

    def choose(self, actions, value_function):
        return random.randint(0, len(actions)-1)

    @property
    def custom_hash(self):
        return __name__+str(self.epsilon)
