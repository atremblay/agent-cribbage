from .register import register
from .value_function import ValueFunction
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
        self.need_training = False

    def forward(self, stacks):
        values = torch.zeros(len(stacks), dtype=torch.float)
        for i, stack in enumerate(stacks.cpu().numpy()):
            myStack = Stack()
            for card_idx in np.argwhere(np.sum(stack, axis=0) == True):
                myStack.add_(Card(*Card.rank_suit_from_idx(int(card_idx))))

            values[i] = torch.tensor(evaluate_table(myStack), dtype=torch.float)

        return values

    def get_after_state(self, state, env):
        choices = np.expand_dims(np.array([c.state for c in state.hand]), axis=1)
        # Store state for data generation.
        after_state = [choices.astype('float32')]
        return after_state