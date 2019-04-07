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
    def stack_and_state_to_numpy(stacks, state, env):
        return [np.array([np.append(p.state, env.dealer == state.hand_id) for p in stacks], dtype=np.float32)]
