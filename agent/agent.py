from .algorithm.register import registry as algo_registry
from .policy.register import registry as policy_registry
from .value_function.register import registry as value_function_registry
from itertools import combinations
from gym_cribbage.envs.cribbage_env import Stack
import numpy as np
import copy
import pickle
import os
import logging


class Agent:
    def __init__(self, algo, policy, value_function):
        self.algo = algo_registry[algo['name']](**algo['kwargs'])
        self.policy = policy_registry[policy['name']](**policy['kwargs'])
        self.value_function = [value_function_registry[value_function['name0']](**value_function['kwargs0']),
                               value_function_registry[value_function['name1']](**value_function['kwargs1'])]
        self.reward = []
        self.cards_2_drop_phase0 = []

        self.data = {'winner': 0, 'data': {0: {}, 1: {}, 2: {}}}
        self._reset_current_data()
        self.logger = logging.getLogger(__name__)

    def _reset_current_data(self):
        self.current_data = [None, None]

    def store_state(self, state):
        if self.current_data[0] is not None:
            raise ValueError('State cannot be overridden.')
        self.current_data[0] = state

    def store_reward(self, reward):
        if self.current_data[1] is None:
            self.current_data[1] = reward
        else:
            self.current_data[1] += reward

        self.reward.append(reward)

    def append_data(self, hand, phase, no_state=False):
        if None in self.current_data and (no_state and self.current_data[1] is not None):
            raise ValueError('Current data cannot be stored since state or reward is None: ' + str(self.current_data))

        this_phase = self.data['data'][phase]
        # Init dictionary for new hand
        if hand not in this_phase:
            this_phase[hand] = [self.current_data]
        else:
            this_phase[hand].append(self.current_data)
        self._reset_current_data()

    def dump_data(self, root, agent_id):
        path = os.path.join(root,
                            str(agent_id)+ '_'+str(self.policy.custom_hash) + '_'
                            + str(self.value_function[0].custom_hash) + '_'
                            + str(self.value_function[1].custom_hash)
                            + '.pickle')
        self.logger.info('Agent '+str(agent_id)+' data saved to :'+path)
        pickle.dump(self.data, open(path, 'wb'))

    @property
    def total_points(self):
        return sum(self.reward)

    def choose(self, state, env):

        choose_phase = [self.choose_phase0, self.choose_phase1]

        return choose_phase[env.phase](state, env)

    def choose_phase0(self, state, env):
        """
        Choose the card to drop in the crib (phase 0). Since the step function plays one card at a time, a buffer is
        created at first step, and then buffer is emptied for subsequent pass. This selection is based on the after
        state (state after dropping cards in the crib).

        :param state:
        :param env:
        :return:
        """

        # If drop buffer is empty (phase 0 begin)
        if len(self.cards_2_drop_phase0) == 0:
            # Unique 4 cards permutations (Good for all numbers of players)
            s_prime_combinations = list(combinations(state.hand, 4))
            # Append dealer to input and convert it to vector state
            S_prime_phase0 = np.array([np.append(Stack(p).state, env.dealer == state.hand_id)
                                       for p in s_prime_combinations])

            # Choose cards to drop according to policy
            after_state = [S_prime_phase0]
            self.store_state(after_state)  # Store state for data generation.
            idx_s_prime = self.policy.choose(after_state, self.value_function[env.phase])

            # Remove cards that stay in hand
            self.cards_2_drop_phase0 = copy.deepcopy(state.hand)
            tuple(self.cards_2_drop_phase0.discard(card) for card in s_prime_combinations[idx_s_prime])

        # Gives next card to drop, and update drop buffer
        card2drop, self.cards_2_drop_phase0 = self.cards_2_drop_phase0[0], self.cards_2_drop_phase0[1:]

        return card2drop

    def choose_phase1(self, state, env):

        hand = np.expand_dims(np.array([c.state for c in state.hand]), axis=1)

        # If has card on the table
        if len(env.table) != 0:
            table_cards = np.expand_dims(np.array([card.state for card in env.table]), 0)
            table_cards_repeated = np.repeat(table_cards, len(state.hand), axis=0)
            hand = np.append(table_cards_repeated, hand, axis=1)

        # Store state for data generation.
        after_state = [hand, np.repeat(np.expand_dims(env.discarded.state, axis=0), len(state.hand), axis=0)]
        self.store_state(after_state)

        idx_s_prime = self.policy.choose(after_state, self.value_function[env.phase])

        return state.hand[idx_s_prime]
