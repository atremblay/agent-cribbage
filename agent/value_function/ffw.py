from .value_function import ValueFunction
import torch.nn as nn
from .register import register


@register
class FFW(nn.Module, ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        # Logistic Regression
        self.ffw = nn.Sequential(
            nn.Linear(52*2, 208),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(208, 1),
            nn.ReLU(True),
        )

        self.apply(self.weights_init)

    def forward(self, x):
        out = self.ffw(x)
        return out




