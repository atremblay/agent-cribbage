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
BATCH_SIZE = 32

Transition = namedtuple('Transition',
    ('state', 'action', 'next_state', 'reward', 'hand'))


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

    def __init__(self, in_dim, out_dim, layers, 
                       compact=False, lr=0.01, device='cpu'):
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

        self.compact = compact  # Determines how cards are represented.

        self.memory = ReplayMemory(REPLAY)

        self.steps = 0

    def get_actions(self, cards, environment, n=1):
        """
        Tests the value of each card in cards. Takes the top n cards in
        sequence, using a given policy.
        """
        actions = []
        values = np.zeros(len(cards))

        for i, card in enumerate(cards):

            # Compact = 13, no suits. Full = 52.
            if self.compact:
                card_state = card.compact_state[1]
            else:
                card_state = card.state

            policy_input = torch.Tensor(np.hstack((card_state, environment)))
            values[i] = self.policy(policy_input.unsqueeze(0))

        for i in range(n):
            action = self.e_greedy(values)
            actions.append(action)
            values[action] = -np.inf  # So this isn't picked next time.

        return(actions)

    def e_greedy(self, values):
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
            return(int(np.argmax(values).astype(np.int)))
        else:
            # Random action, only considering finite values.
            sample = np.random.choice(values[np.isfinite(values)])
            return(int(np.where(values == sample)[0][0]))

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a fial state would've been the one after which simulation ended)       
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
                device=DEVICE, dtype=torch.uint8)

        if torch.sum(non_final_mask) > 0:
            s_prime_batch = torch.stack([s for s in batch.next_state
                                            if s is not None], dim=1)
        else:
            s_prime_batch = None

        # Get states, actions, and rewards.
        s_batch = torch.stack(batch.state, dim=1)
        a_batch = torch.stack(batch.action, dim=1)
        r_batch = torch.stack(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been 
        # taken for each batch state according to policy_net.
        state_action_values = self.policy(
            torch.cat([s_batch, a_batch]).t().float())

        # Compute V(s_{t+1}) for all next states. Final states get 0.
        # Expected values of actions for non_final_next_states are computed
        # based on the older target network.
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

        # Loop through batch... TODO: can this be vectorized?
        for b in range(BATCH_SIZE):
            if non_final_mask[b] == 0:
                continue

            # Get the legal cards for this player.
            hand = batch.hand[b]

            # Take the e-greedy max action.
            action = self.get_actions(hand, s_prime_batch[:, b].numpy())[0]
            a_prime = torch.from_numpy(hand[action].compact_state[1]) 

            # Calculate value of s_prime, a_prime for batch b
            next_state_values[b] = self.target(
                torch.cat([s_prime_batch[:, b], a_prime]).float()).detach()

        # Compute the expected Q values over the decorrelated states.
        expected_state_action_values = (next_state_values * GAMMA) + r_batch.float()

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the policy model.
        self.optimizer.zero_grad()
        loss.backward()

        # Reward clamping ... maybe not a good idea?
        #for param in self.policy.parameters():
        #    param.grad.data.clamp_(-1, 1)
        #    
        self.optimizer.step()

    def target_update(self):
        """Update the target network, copying all weights and biases in DQN."""
        self.target.load_state_dict(self.policy.state_dict())


def get_deal_env_state(env):
    """Environment state during the deal/show."""

    # Are you the dealer? yes=1, no=0
    if env.dealer == env.player:
        dealer_state = np.ones(1)
    else:
        dealer_state = np.zeros(1)

    # What is in your hand? 52
    hand_state = env.hands[env.player].state

    env_state = np.hstack([dealer_state, hand_state])

    return(env_state)


def get_play_env_state(env):
    """Environment state during the play."""
    TABLE_LEN = 7*13

    # Hand state out of 52.
    hand_state = env.hands[env.player].state

    # Sequence of cards on the table: 7*13.
    if env.table:

        # Add compact state of cards in order they were played.
        table_state = []
        for card in env.table:
            table_state.append(card.compact_state[1])
        table_state = np.hstack(table_state)
        
        # Add trailing zeros if the table isn't full.
        if len(table_state) < TABLE_LEN:
            table_state = np.hstack(
                [table_state, np.zeros(TABLE_LEN-len(table_state))])

    else:
        table_state = np.zeros(TABLE_LEN)

    env_state = np.hstack([hand_state, table_state])

    return(env_state)


def get_deal_actions(env, state, Q_deal):
    """
    Given and environment, state, and a policy, select 2 actions.

    Returns an environment state for training Q_deal, and actions_onehot, which
    is part of the state for Q_play.
    """
    env_state = get_deal_env_state(env)
    actions = Q_deal.get_actions(state.hand, env_state, n=2)
    actions_onehot = (
        state.hand[actions[0]].state + state.hand[actions[1]].state)

    return(env_state, actions, actions_onehot)


def main(args):
    """
    deal: S = 6 cards in hand, S' full table with total reward (show).
    play: S = this player's turn, S' = This player's next turn.
    """

    ## Load data
    #kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}
    #train_loader = data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, **kwargs)
    #valid_loader = data.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, **kwargs)

    # Create models
    # Deal: 53 (env_state) + 52 (proposed card).
    Q_deal = QLearner(53+52, 1, [128, 64, 32], lr=args.lr, device=DEVICE)

    # Play: 7*13 (table=7) + 52 (hand) + 13 (proposed card) + 52 (2 crib cards).
    Q_play = QLearner(91+52+13+52, 1, [256, 128, 64, 32], 
                      compact=True, lr=args.lr, device=DEVICE)

    # Init environment.
    env = CribbageEnv()
    state, reward, done, debug = env.reset()

    # Keeps track of the states for each player.

    t1 = time.time()

    done_counter = 1
    step_counter = 1

    while done_counter < 100:

        # S=The deal, S'=The show.
        if state.phase == 0:

            # Initialize player-specific states for this hand.
            hand_sa_deal = [(), ()]
            hand_sar_play = [(), ()]
            hand_cribs = [[], []]
            play_rewards = [0, 0]

            # Based on hand, pick two cards to discard.
            env_state, p1_act, p1_crib = get_deal_actions(env, state, Q_deal)
            hand_sa_deal[state.hand_id] = (env_state, p1_crib)
            p1_discards = [state.hand[p1_act[0]], state.hand[p1_act[1]]]
            hand_cribs[0] = p1_crib
            state, reward, done, debug = env.step(p1_discards[0])
            
            # Based on hand, pick two cards to discard.
            env_state, p2_act, p2_crib = get_deal_actions(env, state, Q_deal)

            hand_sa_deal[state.hand_id] = (env_state, p2_crib)

            p2_discards = [state.hand[p2_act[0]], state.hand[p2_act[1]]]
            hand_cribs[1] = p2_crib
            state, reward, done, debug = env.step(p2_discards[0])

            # Each player discards the other card, and the play is done.
            state, reward, done, debug = env.step(p1_discards[1])           
            s_prime, reward, done, debug = env.step(p2_discards[1])

            # NB: FOR TWO PLAYERS, THE DEAL IS ALWAYS DONE BY NOW.

        # The play. S=player A's turn, S'=player A's next turn.
        elif state.phase == 1:

            env_state = get_play_env_state(env)
            env_state = np.hstack([env_state, hand_cribs[state.hand_id]])

            # Save state, action, reward from previous turn, plus current state
            # of the environment.
            if hand_sar_play[state.hand_id]:
                s, a, r = hand_sar_play[state.hand_id]
                Q_play.memory.push(
                    torch.from_numpy(s), 
                    torch.from_numpy(a), 
                    torch.from_numpy(env_state), 
                    torch.from_numpy(np.array(r)),
                    state.hand)

            action = Q_play.get_actions(state.hand, env_state)
            card_played = state.hand[action[0]]
            s_prime, reward, done, debug = env.step(card_played)

            # Keep track of accumulated rewards during the play.
            play_rewards[s_prime.reward_id] += reward

            hand_sar_play[s_prime.reward_id] = (
                env_state, card_played.compact_state[1], reward)

        # The show.
        elif state.phase == 2:

            # This agent sees what reward they get during the show.
            s_prime, reward, done, debug = env.step(Card(99, SUITS[0]))

            total_play_reward = reward + play_rewards[s_prime.reward_id]

            # Get state, action pair from the deal.
            s, a = hand_sa_deal[state.reward_id]
            Q_deal.memory.push(
                torch.from_numpy(s), 
                torch.from_numpy(a), 
                None, 
                torch.from_numpy(np.array(reward)),
                state.hand)

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