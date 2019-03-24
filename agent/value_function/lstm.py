from .value_function import ValueFunction
import torch.nn as nn
from .register import register


@register
class LSTM(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size=52, hidden_size=104, num_layers=2, batch_first=True)

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(104, 52),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(52, 1),
            nn.ReLU(True),
        )

        self.apply(self.weights_init)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :] # Only keeps last value of sequence
        out = self.clf(out)
        return out




