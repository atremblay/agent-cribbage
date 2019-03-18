from .algorithm.register import registry as algo_registry
from .policy.register import registry as policy_registry


class Agent:
    def __init__(self, algo, policy):
        self.algo = algo_registry[algo['name']](**algo['kwargs'])
        self.policy = policy_registry[policy['name']](**policy['kwargs'])
