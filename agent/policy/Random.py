from .policy import Policy
import random
from .register import register


@register
class Random(Policy):

    def __init__(self):
        super().__init__()

    @staticmethod
    def choose(actions):
        return random.randint(0, len(actions)-1)
