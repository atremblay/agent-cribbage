from .policy import Policy
from .register import register
import random


@register
class Random(Policy):

    def __init__(self):
        super().__init__()

    def choose(self, actions, value_function=None):
        return random.randint(0, len(actions)-1)

    @property
    def custom_hash(self):
        return __name__
