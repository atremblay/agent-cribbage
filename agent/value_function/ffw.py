from .value_function import ValueFunction
import torch.nn as nn
from .register import register


@register
class FFW(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        # Logistic Regression
        self.ffw = nn.Sequential(
            nn.Linear(53, 106),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(106, 1),
            nn.ReLU(True),
        )
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)

    def forward(self, x):
        out = self.ffw(x)
        return out