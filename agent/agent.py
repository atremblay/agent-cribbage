from .algorithm.register import registry as algo_registry
from .policy.register import registry as policy_registry
from .value_function.register import registry as value_function_registry


class Agent:
    def __init__(self, algo, policy, value_function):
        self.algo = algo_registry[algo['name']](**algo['kwargs'])
        self.policy = policy_registry[policy['name']](**policy['kwargs'])
        self.value_function = value_function_registry[value_function['name']](**value_function['kwargs'])
        self.reward = []

    @property
    def total_points(self):
        return sum(self.reward)