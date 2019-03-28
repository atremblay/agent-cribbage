from .value_function import ValueFunction
import torch.nn as nn
from .register import register
import torch
from pathlib import Path
import os
import numpy as np


@register
class Conv(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.suit = nn.Conv1d(
            in_channels=4,
            out_channels=3,
            kernel_size=1,
            stride=1
        )

        self.rank2 = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=2,
            stride=1
        )

        self.rank3 = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=1
        )

        # Logistic Regression
        self.clf = nn.Linear(
            12 * 3 * 3 + 11 * 2 * 3 + 4 * 3,
            1
        )

        self.apply(self.weights_init)

    def forward(self, x_rank, x_suit):
        suit = self.suit(x_suit)
        rank2 = self.rank2(x_rank)
        rank3 = self.rank3(x_rank)

        cat = torch.cat(
            [
                rank2.flatten(start_dim=1),
                rank3.flatten(start_dim=1),
                suit.flatten(start_dim=1)
            ],
            dim=1
        )

        out = self.clf(cat)
        return out


class ConvEval:
    """
    Convenience class to use the value function Conv.
    """

    def __init__(self, path=None, model=None):
        """
        Params
        ======

        path: str
            Path to the folder containing all the training artifacts. It is
            expecting the files model.pkl, kwargs.json and pipeline.pkl

        model: torch.nn.Module
        """
        super(ConvEval, self).__init__()
        if path is not None:
            self.path = Path(path)

            self.model = Conv()
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.path, 'model.pkl'),
                    map_location='cpu'
                )
            )
        else:
            self.model = model
        self.model.eval()

    def predict(self, stack):
        if len(stack) != 4:
            raise Exception("Expecting the stack to have 4 cards")

        ranks = np.zeros((1, 1, 13, 4), dtype=np.float32)
        suits = np.zeros((1, 4, 4), dtype=np.float32)
        s, r = stack.compact_state
        ranks[0, 0] = r
        suits[0] = s

        return self.model(torch.tensor(ranks), torch.tensor(suits))
