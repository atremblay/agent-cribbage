import torch
import torch.nn as nn

from ...utils.device import device


class ValueFunction(nn.Module):
    def __init__(self):
        """
        """
        super().__init__()
        self.need_training = True

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def update(self, batch):
        pass

    def evaluate(self, xs):
        self.eval()
        xs = [device(torch.tensor(x, dtype=torch.float)) for x in xs]
        return self.forward(*xs)