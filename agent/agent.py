from .algorithm.register import registry as algo_registry
from .policy.register import registry as policy_registry
from .value_function.register import registry as value_function_registry

from itertools import combinations
from gym_cribbage.envs.cribbage_env import Stack
import numpy as np
import copy


class Agent:
    def __init__(self, algo, policy, value_function):
        self.algo = algo_registry[algo['name']](**algo['kwargs'])
        self.policy = policy_registry[policy['name']](**policy['kwargs'])
        self.value_function = [value_function_registry[value_function['name0']](**value_function['kwargs0']),
                               value_function_registry[value_function['name1']](**value_function['kwargs1'])]
        self.reward = []
        self.cards_2_drop_phase0 = []

    @property
    def total_points(self):
        return sum(self.reward)

    def choose(self, state, env):

        choose_phase = [self.choose_phase0, self.choose_phase1]

        return choose_phase[env.phase](state, env)

    def choose_phase0(self, state, env):
        if len(self.cards_2_drop_phase0) == 0:
            # Unique 4 cards permutations (Good for all numbers of players)
            s_prime_combinations = list(combinations(state.hand, 4))
            S_prime_phase0 = np.array([np.append(Stack(p).state, env.dealer == state.hand_id) for p in s_prime_combinations])

            idx_s_prime = self.policy.choose(S_prime_phase0, self.value_function[env.phase])

            self.cards_2_drop_phase0 = copy.deepcopy(state.hand)
            #Remove cards that stay in hand
            tuple(self.cards_2_drop_phase0.discard(card) for card in s_prime_combinations[idx_s_prime])

        card2drop, self.cards_2_drop_phase0 = self.cards_2_drop_phase0[0], self.cards_2_drop_phase0[1:]

        return card2drop

    def choose_phase1(self, state, env):
        idx_s_prime = policy_registry['Random']().choose(np.array([c.state for c in state.hand]), None)
        return state.hand[idx_s_prime]
