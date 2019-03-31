from .value_function import ValueFunction
from .register import register
from gym_cribbage.envs.cribbage_env import evaluate_table, Card, Stack
import numpy as np
import torch


@register
class EvalPlays(ValueFunction):
    def __init__(self, out_channels=15):
        """
        """
        super().__init__()
        self.custom_hash = __name__ + 'V0.0.0'

    def forward(self, stacks, discarded):
        values = torch.zeros(len(stacks), dtype=torch.float)
        for i, stack in enumerate(stacks.numpy()):
            myStack = Stack()
            for card_idx in np.argwhere(np.sum(stack, axis=0) == True):
                myStack.add_(Card(*Card.rank_suit_from_idx(int(card_idx))))

            values[i] = torch.tensor(evaluate_table(myStack), dtype=torch.float)

        return values