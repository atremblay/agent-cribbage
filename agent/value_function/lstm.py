from .value_function import ValueFunction
import torch.nn as nn
from .register import register
import torch
from gym_cribbage.envs.cribbage_env import Stack, RANKS, SUITS
import numpy as np


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
            nn.Linear(104+52, 52),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(52, 1),
            nn.ReLU(True),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x, discarded):

        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]  # Only keeps last value of sequence
        out = self.clf(torch.cat((out, discarded), dim=1))
        return out


@register
class SimpleLSTM(ValueFunction):
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
            input_size=10,
            hidden_size=104,
            num_layers=2,
            batch_first=True
        )

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(104, 52),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(52, 1),
            nn.ReLU(True),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x):
        # out = self.pad(x.transpose(1, 2)).transpose(1, 2)
        out2 = self.conv2(x.unsqueeze(1)).sum(dim=-1)
        out3 = self.conv3(x.unsqueeze(1)).sum(dim=-1)
        out = torch.cat(
            [out2.transpose(1, 2), out3.transpose(1, 2)],
            dim=2
        )
        out, (hidden, cell) = self.lstm(out)
        out = out[:, -1, :]  # Only keeps last value of sequence
        out = self.clf(out)
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
