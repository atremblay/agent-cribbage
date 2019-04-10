from .policy.register import registry as policy_registry
from .value_function.register import registry as value_function_registry
from gym_cribbage.envs.cribbage_env import Stack, evaluate_cards
from itertools import combinations
import copy
import logging
import os
import pickle
import torch


class Agent:
    def __init__(self, policies, value_functions, optimizers, algorithms, training_settings=None):

        self.policies = [policy_registry[p['class']](**p['kwargs']) for p in policies]
        self.value_functions = [value_function_registry[v['class']](**v['kwargs']) for v in value_functions]
        self.algorithms = algorithms
        self.optimizers_define = optimizers
        self.optimizers = []
        self.scheduler = []
        self.choose_phase = [getattr(self, p['callback']['name']) for p in policies]
        self.choose_phase_kwargs = [p['callback']['kwargs'] for p in policies]
        self.training_settings = training_settings

        self.cards_2_drop_phase0 = []
        self.reset()
        self.data = {'winner': 0, 'data': {0: {}, 1: {}, 2: {}}}
        self._reset_current_data()
        self.logger = logging.getLogger(__name__)

    def reset(self):
        self.reward = []

    def _reset_current_data(self):
        self.current_data = [None, None]

    def store_state(self, state, idx_choice):
        if self.current_data[0] is not None:
            raise ValueError('State cannot be overridden.')
        self.current_data[0] = (state, idx_choice)

    def store_reward(self, reward):
        if self.current_data[1] is None:
            self.current_data[1] = reward
        else:
            self.current_data[1] += reward

        self.reward.append(reward)

    def append_data(self, hand_no, phase, no_state=False):
        if None in self.current_data and (no_state and self.current_data[1] is not None):
            raise ValueError('Current data cannot be stored since state or reward is None: ' + str(self.current_data))

        this_phase = self.data['data'][phase]
        # Init dictionary for new hand
        hand = 'hand-' + str(hand_no)
        if hand not in this_phase:
            this_phase[hand] = [self.current_data]
        else:
            this_phase[hand].append(self.current_data)
        self._reset_current_data()

    def dump_data(self, root, agent_id):
        path = os.path.join(root, agent_id+'_'+self.hash()+'.pickle')
        self.logger.debug('Agent '+agent_id+' data saved to :'+path)
        pickle.dump(self.data, open(path, 'wb'))
        self.data = {'winner': 0, 'data': {0: {}, 1: {}, 2: {}}} # Reset data
        return path

    def hash(self):
        return '_'.join([str(v.custom_hash) for v in self.policies]+[str(v.custom_hash) for v in self.value_functions])

    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.init_optimizer()
        for v, v_state in zip(self.value_functions, checkpoint['model_state_dict']):
            v.load_state_dict(v_state)

        return checkpoint['epoch']

    def save_checkpoint(self, checkpoint_file, epoch):
        model_state_dict = []
        for v in self.value_functions:
            model_state_dict.append(v.state_dict())

        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
        }, checkpoint_file)

    @property
    def total_points(self):
        return sum(self.reward)

    def init_optimizer(self):
        for i, o in enumerate(self.optimizers_define):
            if self.value_functions[i].need_training:
                opt = getattr(torch.optim, o['class'])(self.value_functions[i].parameters(), **o['kwargs'])
                scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
                self.optimizers.append(opt)
                self.scheduler.append(scheduler)
            else:
                self.optimizers.append(None)
                self.scheduler.append(None)

    def choose(self, state, env):
        return self.choose_phase[env.phase](state, env, **self.choose_phase_kwargs[env.phase])

    def human_input(self, hand):
        while 1:
            self.logger.human('\n' + 'Idx:\t\t'+ '\t'.join([str(j)+''.join([' ' for _ in range(len(str(c))-1)]) for j, c in enumerate(hand)]))
            self.logger.human('Your hand: \t'+'\t'.join([str(c) for c in hand]))
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

                    self.logger.human('\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
                                      '!!!!!!You are the dealer!!!!!')
                else:
                    self.logger.human('\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
                                      '!!!!!You are NOT the dealer!!!!!')

                temp_hand = copy.deepcopy(state.hand)
                stack = Stack()
                while len(temp_hand) > 4:
                    card = self.human_input(temp_hand)
                    stack.add_(card)
                    temp_hand.discard(card)

                self.cards_2_drop_phase0 = stack

            else:
                # Unique 4 cards permutations (Good for all numbers of players)
                s_prime_combinations = [Stack(c) for c in combinations(state.hand, 4)]
                # Convert stack to numpy
                after_state = self.value_functions[env.phase].get_after_state(s_prime_combinations, state, env)

                # Store state for data generation.
                idx_s_prime = self.policies[env.phase].choose(after_state, self.value_functions[env.phase])
                self.store_state(after_state, idx_s_prime)

                # Remove cards that stay in hand
                self.cards_2_drop_phase0 = copy.deepcopy(state.hand)
                tuple(self.cards_2_drop_phase0.discard(card) for card in s_prime_combinations[idx_s_prime])

        # Gives next card to drop, and update drop buffer
        card2drop, self.cards_2_drop_phase0 = self.cards_2_drop_phase0[0], self.cards_2_drop_phase0[1:]

        return card2drop

    def choose_phase1(self, state, env, human=False):

        spacing = '\n\n====================================\n'
        if env.prev_phase != 1:
            self.logger.human(spacing+'Starter: ' + str(env.starter))
            spacing = ''

        if human:

            self.logger.human(spacing+'Table: '+str(env.table)+' (count= '+str(sum([c.value for c in env.table]))+')')
            return self.human_input(state.hand)

        else:

            after_state = self.value_functions[env.phase].get_after_state(state, env)
            idx_s_prime = self.policies[env.phase].choose(after_state, self.value_functions[env.phase])
            self.store_state(after_state, idx_s_prime)

            return state.hand[idx_s_prime]

    def choose_random(self, state, env):
        idx_s_prime = self.policies[env.phase].choose(state.hand)
        return state.hand[idx_s_prime]

