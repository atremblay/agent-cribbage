from .value_function import ValueFunction
from .register import register
from gym_cribbage.envs.cribbage_env import evaluate_cards, Card, Stack
import numpy as np
import torch

@register
class EvalCards(ValueFunction):
    def __init__(self, out_channels=15):
        """
        """
        super().__init__()
        self.custom_hash = __name__ + 'V0.0.0'

    @staticmethod
    def stack_to_numpy(stacks, state, env):
        return [np.array([p.state for p in stacks])]

    def forward(self, stacks):
        values = torch.zeros(len(stacks), dtype=torch.float)
        for i, stack in enumerate(stacks.numpy()):
            myStack = Stack()
            for card_idx in np.argwhere(stack == True):
                myStack.add_(Card(*Card.rank_suit_from_idx(int(card_idx))))

            values[i] = torch.tensor(evaluate_cards(myStack), dtype=torch.float)

        return values