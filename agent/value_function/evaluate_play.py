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
            assert (stack.sum(axis=1) == 1).all()
            for card_idx in np.argmax(stack, axis=1):
                myStack.add_(Card(*Card.rank_suit_from_idx(int(card_idx))))

            values[i] = torch.tensor(evaluate_table(myStack), dtype=torch.float)

        return values

    def get_after_state(self, state, env):
        after_state = np.zeros(
            (len(state.hand), len(env.table) + 1, 52),
            dtype=np.float32
        )
        for i, card in enumerate(state.hand):
            table = env.table.add(card)
            for j, card_on_table in enumerate(table):
                after_state[i, j] = card_on_table.state

        return [after_state]
