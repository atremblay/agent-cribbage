from .policy.register import registry as policy_registry
from .value_function.register import registry as value_function_registry
from itertools import combinations
from gym_cribbage.envs.cribbage_env import Stack
import numpy as np
import pickle
import os
import logging
import copy


class Agent:
    def __init__(self, policies, value_functions):

        self.policies = [policy_registry[p['class']](**p['kwargs']) for p in policies]
        self.value_functions = [value_function_registry[v['class']](**v['kwargs']) for v in value_functions]
        self.choose_phase = [getattr(self, p['callback']['name']) for p in policies]
        self.choose_phase_kwargs = [p['callback']['kwargs'] for p in policies]

        self.reward = []
        self.cards_2_drop_phase0 = []

        self.data = {'winner': 0, 'data': {0: {}, 1: {}, 2: {}}}
        self._reset_current_data()
        self.logger = logging.getLogger(__name__)

    def reset(self):
        self.reward = []

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
        path = os.path.join(root, agent_id+'_'+self.hash()+'.pickle')
        self.logger.info('Agent '+agent_id+' data saved to :'+path)
        pickle.dump(self.data, open(path, 'wb'))
        return path

    def hash(self):
        return '_'.join([str(v.custom_hash) for v in self.policies]+[str(v.custom_hash) for v in self.value_functions])

    @property
    def total_points(self):
        return sum(self.reward)

    def choose(self, state, env):

        return self.choose_phase[env.phase](state, env, **self.choose_phase_kwargs[env.phase])

    def human_input(self, hand):
        while 1:
            self.logger.human('\n' + str(hand))
            try:
                idx_card = int(input('Select index of card to play on the table: '))
                card = hand[idx_card]
                self.logger.human('Play: ' + str(card))
                return card
            except Exception as e:
                self.logger.human('ERROR: Invalid Input -> '+str(e))


    def choose_phase0(self, state, env, human=False):
        """
        Choose the card to drop in the crib (phase 0). Since the step function plays one card at a time, a buffer is
        created at first step, and then buffer is emptied for subsequent pass. This selection is based on the after
        state (state after dropping cards in the crib).

        :param state:
        :param env:
        :param human:
        :return:
        """

        # If drop buffer is empty (phase 0 begin)
        if len(self.cards_2_drop_phase0) == 0:

            if human:
                if env.dealer == env.player:
                    self.logger.human('\n\n\n!!!!!!You are the dealer!!!!!')
                else:
                    self.logger.human('\n\n\n!!!!!You are NOT the dealer!!!!!')

                temp_hand = copy.deepcopy(state.hand)
                stack = Stack()
                while len(temp_hand) > 4:
                    card = self.human_input(temp_hand)
                    stack.add_(card)
                    temp_hand.discard(card)

                self.cards_2_drop_phase0 = stack

            else:
                # Unique 4 cards permutations (Good for all numbers of players)
                s_prime_combinations = list(combinations(state.hand, 4))
                # Append dealer to input and convert it to vector state
                S_prime_phase0 = np.array([np.append(Stack(p).state, env.dealer == state.hand_id)
                                           for p in s_prime_combinations])

                # Choose cards to drop according to policy
                after_state = [S_prime_phase0]
                self.store_state(after_state)  # Store state for data generation.
                idx_s_prime = self.policies[env.phase].choose(after_state, self.value_functions[env.phase])

                # Remove cards that stay in hand
                self.cards_2_drop_phase0 = copy.deepcopy(state.hand)
                tuple(self.cards_2_drop_phase0.discard(card) for card in s_prime_combinations[idx_s_prime])

        # Gives next card to drop, and update drop buffer
        card2drop, self.cards_2_drop_phase0 = self.cards_2_drop_phase0[0], self.cards_2_drop_phase0[1:]

        return card2drop

    def choose_phase1(self, state, env, human=False):

        if human:
            self.logger.human('\n\nStarter: '+str(env.starter))
            self.logger.human('Table: '+str(env.table)+' (count= '+str(sum([c.value for c in env.table]))+')')
            return self.human_input(state.hand)

        else:
            hand = np.expand_dims(np.array([c.state for c in state.hand]), axis=1)

            # If has card on the table
            if len(env.table) != 0:
                table_cards = np.expand_dims(np.array([card.state for card in env.table]), 0)
                table_cards_repeated = np.repeat(table_cards, len(state.hand), axis=0)
                hand = np.append(table_cards_repeated, hand, axis=1)

            # Store state for data generation.
            after_state = [hand, np.repeat(np.expand_dims(env.discarded.state, axis=0), len(state.hand), axis=0)]
            self.store_state(after_state)

            idx_s_prime = self.policies[env.phase].choose(after_state, self.value_functions[env.phase])

            return state.hand[idx_s_prime]

    def choose_random(self, state, env):
        idx_s_prime = self.policies[env.phase].choose(state.hand)
        return state.hand[idx_s_prime]
