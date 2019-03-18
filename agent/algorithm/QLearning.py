from .algorithm import Algorithm
from .register import register


@register
class QLearning(Algorithm):
    def __init__(self):
        super(QLearning).__init__()