from .register import register
from .value_function import ValueFunction
from gym_cribbage.envs.cribbage_env import Stack, RANKS, SUITS
import numpy as np
import torch
import torch.nn as nn


@register
class DeeperFF(ValueFunction):
    def __init__(self, num_features, n_hid1=10, n_hid2=5, n_hid3=2):
        super().__init__()
        self.model = nn.Sequential(

            nn.Linear(num_features, n_hid1),
            nn.ReLU(),

            nn.Linear(n_hid1, n_hid2),
            nn.ReLU(),

            nn.Linear(n_hid2, n_hid3),
            nn.ReLU(),

            nn.Linear(n_hid3, 1),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Only good for visual??
                nn.init.constant_(m.bias, 0)
                
    def forward(self, input_tensor):
        return self.model(input_tensor)

    def get_after_state(self, state, env):

        # Sort hand

        hand = [c.rank_value for c in h]
        hand.sort()
        for i, c in enumerate(hand):
            data_mat[i * 13 + c - 1, j] = 1
        for i, c in enumerate(t):
            i += 4
            data_mat[i * 13 + c.rank_value - 1, j] = 1

        choices = np.expand_dims(np.array([c.state for c in state.hand]), axis=1)


        # Store state for data generation.
        after_state = [choices.astype('float32')]


        return after_state

