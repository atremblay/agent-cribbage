#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
from gym_cribbage.envs.cribbage_env import CribbageEnv, Card, RANKS, SUITS
from models import Linear
from torch.utils import data
import argparse
import math
import numpy as np
import random
import sys
import time
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
REPLAY = 10000
GAMMA = 1

Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward'))

# class FeatRepDataset(data.Dataset):
#     """
#     This is a class working with Feature Representation of cards.
#     Inputs is 13*12, reprensenting 12 cards (1 to 13). 0-4 Hand, 5-12 Table
#     Target is the max reward in this setting
#
#     Returns a sample (input) and a targeted value (target).
#     """
#
#     def __init__(self, d, t):
#         self.d = d
#         self.t = t
#
#     def __len__(self):
#         "Total number of samples."
#         return len(self.t)
#
#     def __getitem__(self, index):
#         "Generate one sample of data."
#         return self.d[:,index], self.t[index]

#def to_card(data_mat):
#    """ Function just to see human way of hand """
#
#    cards = []
#    for i in range(12):
#        c = (data_mat[i*13:i*13+13]).nonzero()+1
#        if c.nelement() != 0: c = c[0].item()
#        else: c = 0
#        cards.append(c)
#
#    return cards


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearner(object):

    def __init__(self, in_dim, out_dim, layers, lr=0.01, device='cpu'):
        """Init policy network, target network, and optimizer."""

        assert isinstance(layers, list)

        self.policy = Linear(in_dim, out_dim, layers)
        self.policy.train()

        self.optimizer = torch.optim.SGD(
            self.policy.parameters(), lr=lr, momentum=0.9, nesterov=True)

        self.target = Linear(in_dim, out_dim, layers)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.criterion = nn.SmoothL1Loss()  # Huber Loss

        self.memory = ReplayMemory(REPLAY)

        self.steps = 0

    def get_actions(cards, environment, n=1):
        """
        Tests the value of each card in cards. Takes the top n cards in
        sequence, using a given policy.
        """
        actions = []
        values = np.zeros(len(cards))

        for i, card in enumerate(cards):
            values[i] = self.policy(card+environment)

        for i in range(n):
            action = self.e_greedy(values)
            actions.append(action)
            values[action] = -np.inf  # So this isn't picked next time.

        return(actions)

    def e_greedy(values):
        """
        1-EPS of the time, returns the argmax. Else, random. EPS decays
        over time.
        """
        sample = random.random()
        threshold = (EPS_END + (EPS_START - EPS_END)
                     * math.exp(-1. * self.steps/EPS_DECAY))
        self.steps += 1

        if sample > threshold:
            # Greedy action.
            return(np.argmax(values))
        else:
            # Random action.
            return(np.where(values == np.random.selection(values))[0])

    def optimize():
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        s_batch = torch.cat(batch.state)
        a_batch = torch.cat(batch.action)
        r_batch = torch.cat(batch.reward)
        s_prime_batch = torch.cat(batch.next_state)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #non_final_mask = torch.tensor(
        #    tuple(map(lambda s: s is not None, batch.next_state)),
        #        device=device, dtype=torch.uint8)
        #non_final_next_states = torch.cat([s for s in batch.next_state
        #                                            if s is not None])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy(s_batch).gather(1, a_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed
        # based on the older target network; selecting their best reward with
        # max(1)[0].
        #next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values = self.target(s_prime_batch).max(1)[0].detach()

        # Compute the expected Q values over the decorrelated states.
        expected_state_action_values = (next_state_values * GAMMA) + r_batch

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the policy model.
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def target_update():
        """Update the target network, copying all weights and biases in DQN."""
        self.target.load_state_dict(self.policy.state_dict())


def main():
    """
    deal: S = 6 cards in hand, S' full table with total reward (show).
    play: S = this player's turn, S' = This player's next turn.
    """

    ## Load data
    #kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    #train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, **kwargs)
    #valid_loader = data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, **kwargs)

    # Create models
    # Deal: 52*6 + Dealer (1), softmax over 6 cards.
    # Play: 12*13 (hand=4, table=8), suits do not matter, softmax over 4 cards.
    Q_deal = QLearner(313, 1, [256, 128, 64], lr=args.lr, device=DEVICE)
    Q_play = QLearner(156, 1, [128, 64, 32], lr=args.lr, device=DEVICE)

    # Init environment.
    env = CribbageEnv()
    state, reward, done, debug = env.reset()

    # Keeps track of the states for each player.
    player_sa_deal = [(), ()]
    player_sar_play = [(), ()]

    t1 = time.time()

    done_counter = 1
    step_counter = 1

    while done_counter < 100:

        player = state.hand_id
        print(player)

        # S=The deal, S'=The show.
        if state.phase == 0:

            # Each player discards two cards to the crib in sequence.
            # We save as the state each player's hand.

            # Based on hand, pick two cards to discard.
            env_state = env.table + env.crib + env.dealer
            full_state = state.hand.state + env_state
            p1_actions = Q_deal.get_actions(state.hand.state, env_state, n=2)
            player_sa_deal[player] = (full_state, p1_actions)
            state, reward, done, debug = env.step(state.hand[p1_actions[0]])

            # Based on hand, pick two cards to discard.
            player = state.hand_id
            env_state = env.table + env.crib + env.dealer
            full_state = state.hand.state + env_state
            p2_actions = Q_deal.get_actions(state.hand.state, env_state, n=2)
            player_sa_deal[player] = (full_state, p2_actions)
            state, reward, done, debug = env.step(state.hand[p2_actions[0]])

            # Each player discards the other card, and the play is done.
            state, reward, done, debug = env.step(state.hand[p1_actions[1]])
            s_prime, reward, done, debug = env.step(state.hand[p2_actions[1]])

        # The play. S=player A's turn, S'=player A's next turn.
        elif state.phase == 1:

            # Save state, action, reward from previous turn, plus current state
            # of the environment.
            if player_sar_play[player]:
                s, a, r = player_sar_play[player]
                Q_play.memory.push(s, a, state, r)

            action = Q_play.get_actions(state.hand.state, env_state)
            s_prime, reward, done, debug = env.step(state.hand[action])
            player_sar_play[player] = (state, action, reward)

        # The show.
        elif state.phase == 2:
            import IPython; IPython.embed()

            # This agent sees what reward they get during the show.
            s_prime, reward, done, debug = env.step(Card(RANKS[0], SUITS[0]))

            # Get state, action pair from the play.
            s, a = player_sa_deal[player]
            Q_deal.memory.push(s, a, state, reward)

            show_state = env.table.state

        else:
            raise Exception("state.phase is an illegal value={}".format(
                state.phase))

        # Optimize models.
        Q_deal.optimize()
        Q_play.optimize()

        step_counter += 1

        state = s_prime

        if step_counter % TARGET_UPDATE == 0:
            Q_deal.target_update()
            Q_play.target_update()

        # Full game is done (a player hit 121).
        if done:
            env.render()
            state, reward, done, debug  = env.reset()
            done_counter += 1

    t2 = time.time()

    print("100 games in {} seconds".format(t2-t1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TD lambda for Cribbage')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=1,
                        help='discount')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='number of episodes')
    parser.add_argument('--patience', type=int, default=5,
                        help='n valid epochs of no improvement before end.')
    args = parser.parse_args()

    main(args)