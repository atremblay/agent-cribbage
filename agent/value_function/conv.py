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
    def __init__(self, out_channels=15):
        """
        """
        super().__init__()

        self.tarot2 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=2,
                stride=1
            ),
            nn.ReLU()
        )
        num_outputs = out_channels * 12
        self.tarot3 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=3,
                stride=1
            ),
            nn.ReLU()
        )
        num_outputs += out_channels * 11
        self.tarot4 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=4,
                stride=1
            ),
            nn.ReLU()
        )
        num_outputs += out_channels * 10
        self.suit = nn.Conv1d(
            in_channels=1,
            out_channels=5,
            kernel_size=1,
            stride=1
        )
        num_outputs += 4 * 5
        num_outputs += 13

        self.ranks = nn.Sequential(
            nn.Linear(13, 13),
            nn.Tanh(),
            nn.Linear(13, 13)
        )

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(
                num_outputs,
                num_outputs
            ),
            nn.ReLU(),
            nn.Linear(
                num_outputs,
                num_outputs
            ),
            nn.ReLU(),
            nn.Linear(num_outputs, 1)
        )

        self.apply(self.weights_init)
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed

    @staticmethod
    def stack_to_tensor(stacks):
        return torch.tensor(stack_to_numpy(stacks))

    @staticmethod
    def stack_to_numpy(stacks, state, env):
        return stack_to_numpy(stacks)

    def forward(self, x_tarot):
        suit = self.suit(x_tarot.sum(dim=2).sum(dim=-1).unsqueeze(1))
        tarot = x_tarot.sum(dim=1).sum(dim=-1).unsqueeze(1)
        tarot2 = self.tarot2(tarot)
        tarot3 = self.tarot3(tarot)
        tarot4 = self.tarot4(tarot)

        cat = torch.cat(
            [
                tarot2.flatten(start_dim=1),
                tarot3.flatten(start_dim=1),
                tarot4.flatten(start_dim=1),
                self.ranks(x_tarot.sum(dim=1).sum(dim=-1)),
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

        tarot = Conv.stack_to_tensor(stack)

        return self.model(torch.tensor(tarot))


def stack_to_numpy(stacks):
    if isinstance(stacks, Stack):
        stacks = [stacks]
    batch_size = len(stacks)
    tarot = np.zeros((batch_size, 4, 13, 4), dtype=np.float32)
    for i, stack in enumerate(stacks):
        for j, card in enumerate(sorted(stack)):
            rank, suit = RANKS.index(card.rank), SUITS.index(card.suit)
            tarot[i, suit, rank, j] = 1
    return tarot