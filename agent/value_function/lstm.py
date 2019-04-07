from .register import register
from .value_function import ValueFunction
from gym_cribbage.envs.cribbage_env import Stack, RANKS, SUITS
import numpy as np
import torch
import torch.nn as nn


@register
class LSTM(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=52,
            hidden_size=104,
            num_layers=2,
            batch_first=True
        )

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(104+52+13, 52),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(52, 1),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x, discarded, hand):

        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]  # Only keeps last value of sequence
        out = self.clf(torch.cat((out, discarded, hand), dim=1))
        return out

    def get_after_state(self, state, env):
        choices = np.expand_dims(np.array([c.state for c in state.hand]), axis=1)

        # If has card on the table
        if len(env.table) != 0:
            table_cards = np.expand_dims(np.array([card.state for card in env.table]), 0)
            table_cards_repeated = np.repeat(table_cards, len(state.hand), axis=0)
            choices = np.append(table_cards_repeated, choices, axis=1)

        hand = np.array([state.hand.remove(c).compact_state[1].sum(axis=1) for c in state.hand])

        # Store state for data generation.
        after_state = [choices.astype('float32'),
                       np.repeat(np.expand_dims(env.discarded.state, axis=0), len(state.hand), axis=0).astype(
                           'float32'),
                       hand]

        return after_state


@register
class ConvLstm(ValueFunction):
    def __init__(self):
        """
        Simple LSTM that only takes the cards face up. We do not include
        face down here, thereby reducing the number of possible states.
        """
        super().__init__()

        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=5,
                kernel_size=2,
                stride=1
            )
        )

        self.conv3 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 2, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=5,
                kernel_size=3,
                stride=1
            )
        )

        self.lstm = nn.LSTM(
            input_size=10+13,
            hidden_size=104,
            num_layers=2,
            batch_first=True
        )

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(104+52+13, 52),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(52, 1),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x, discarded, hand):
        # out = self.pad(x.transpose(1, 2)).transpose(1, 2)
        out2 = self.conv2(x.unsqueeze(1)).sum(dim=-1)
        out3 = self.conv3(x.unsqueeze(1)).sum(dim=-1)
        out = torch.cat(
            [out2.transpose(1, 2), out3.transpose(1, 2), x],
            dim=2
        )
        out, (hidden, cell) = self.lstm(out)
        out = out[:, -1, :]  # Only keeps last value of sequence
        out = self.clf(torch.cat((out, discarded, hand), dim=1))
        return out

    @staticmethod
    def stack_to_numpy(stacks):
        if isinstance(stacks, Stack):
            stacks = [stacks]

        max_len = max([len(s) for s in stacks])
        batch_size = len(stacks)
        x = np.zeros((batch_size, max_len, 13), dtype=np.float32)

        for i, stack in enumerate(stacks):
            for j, card in enumerate(sorted(stack, reverse=True)):
                rank = RANKS.index(card.rank)
                x[i, -(j + 1), rank] = 1
        return x

    def get_after_state(self, state, env):
        choices = [env.table.add(c) for c in state.hand]

        choices = self.stack_to_numpy(choices)

        hand = np.array([state.hand.remove(c).compact_state[1].sum(axis=1) for c in state.hand])

        # Store state for data generation.
        after_state = [choices,
                       np.repeat(np.expand_dims(env.discarded.state, axis=0), len(state.hand), axis=0).astype(
                           'float32'),
                       hand]

        return after_state
