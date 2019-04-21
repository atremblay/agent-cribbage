from .register import register
from .value_function import ValueFunction
import numpy as np
import torch.nn as nn


@register
class FFW(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        # Logistic Regression
        self.ffw = nn.Sequential(
            nn.Linear(53, 106),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(106, 1),
            nn.ReLU(),
        )
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x):
        out = self.ffw(x)
        return out

    @staticmethod
    def get_after_state(state, env):
        return [np.array([np.append(p.state, env.dealer == state.hand_id) for p in state.hand], dtype=np.float32)]


@register
class DeeperFF(ValueFunction):
    def __init__(self, num_features=143, n_hid1=128, n_hid2=64, n_hid3=32):
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
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed

    def forward(self, input_tensor):
        return self.model(input_tensor)

    def get_after_state(self, state, env):

        frs = np.zeros((len(state.hand), 11 * 13))

        for i, card in enumerate(state.hand):
            frs[i] = self.feature_representation_Full(state, env, card)

        return [frs]

    @staticmethod
    def feature_representation_Full(state, env, action):
        """
        Turns a hand, env, action in it's
        Full feature representation for Q, i.e. a vector of (143):
            13 (Hand)
            + 13 (player's cards played, starter, player's cards in crib)
            + 13 (cards played by opponent)
            + 7*13 (91) for ordered table
            + 13 Action, card to play
        """

        fr = np.zeros(11 * 13)

        # Hand
        hand = [c.rank_value for c in state.hand]
        for c in (hand):
            fr[c - 1] += 1

        # Played by opponent
        played_opp = [c.rank_value for c in env.played[int(not state.hand_id)]]
        for c in (played_opp):
            fr[13 + c - 1] += 1

        # Cards not available anymore
        # Get crib of player
        if state.hand_id == env.dealer:
            crib = [env.crib[0]] + [env.crib[2]]
        else:
            crib = [env.crib[1]] + [env.crib[3]]
        not_avail = [c.rank_value for c in env.played[state.hand_id]] + \
                    [c.rank_value for c in env.starter] + \
                    [c.rank_value for c in crib]
        for c in (not_avail):
            fr[2 * 13 + c - 1] += 1

            # Table
        for i, c in enumerate(env.table):
            i += 3
            fr[i * 13 + c.rank_value - 1] = 1

        # Action
        fr[10 * 13 + action.rank_value - 1] = 1

        return fr

