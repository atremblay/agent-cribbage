from .value_function import ValueFunction
import torch.nn as nn
from .register import register
import torch
from pathlib import Path
import os
import numpy as np
from gym_cribbage.envs.cribbage_env import Stack, RANKS, SUITS


@register
class Conv(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.tarot = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=15,
                kernel_size=4,
                stride=1
            ),
            nn.ReLU()
        )

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(
                1 * 15 * 10 + 10,
                100
            ),
            nn.Tanh(),
            nn.Linear(100, 1)
        )

        self.apply(self.weights_init)

    @staticmethod
    def stack_to_tensor(stacks):
        if isinstance(stacks, Stack):
            stacks = [stacks]
        batch_size = len(stacks)
        ranks = np.zeros((batch_size, 10), dtype=np.float32)
        tarot = np.zeros((batch_size, 4, 13, 4), dtype=np.float32)
        for i, stack in enumerate(stacks):
            for j, card in enumerate(sorted(stack)):
                rank, suit = RANKS.index(card.rank), SUITS.index(card.suit)
                tarot[i, suit, rank, j] = 1
                ranks[i, card.value - 1] += 1
        tarot = torch.tensor(tarot)
        ranks = torch.tensor(ranks)
        return ranks, tarot

    def forward(self, x_rank, x_tarot):
        tarot = self.tarot(x_tarot)

        cat = torch.cat(
            [
                tarot.flatten(start_dim=1),
                x_rank
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

        ranks, tarot = Conv.stack_to_tensor(stack)

        return self.model(torch.tensor(ranks), torch.tensor(tarot))
