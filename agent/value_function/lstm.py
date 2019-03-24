from .value_function import ValueFunction
import torch.nn as nn
from .register import register


@register
class LSTM(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size=104, hidden_size=208, num_layers=2)

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(208, 416),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(416, 1),
            nn.ReLU(True),
        )

        self.apply(self.weights_init)

    def forward(self, x):
        out = self.lstm(x)
        out = self.clf(out.view(out.size(0), -1))
        return out




