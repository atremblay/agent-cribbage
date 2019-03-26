from .value_function import ValueFunction
import torch.nn as nn
from .register import register
import torch

@register
class LSTM(ValueFunction):
    def __init__(self):
        """
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size=52, hidden_size=104, num_layers=2, batch_first=True)

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(104+52, 52),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(52, 1),
            nn.ReLU(True),
        )
        # Before applying weights
        self.custom_hash = __name__ + 'V0.0.0'  # Change version when network is changed
        self.apply(self.weights_init)



    def forward(self, x, discarded):

        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]  # Only keeps last value of sequence
        out = self.clf(torch.cat((out, discarded), dim=1))
        return out





