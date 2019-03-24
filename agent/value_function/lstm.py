from .value_function import ValueFunction
import torch.nn as nn
from .register import register


@register
class LSTM(nn.Module, ValueFunction):
    def __init__(self):
        """

        :param in_channels:
        :param down_blocks:
        :param up_blocks:
        :param bottleneck_layers:
        :param growth_rate:
        :param out_chans_first_conv:
        :param n_classes: If n_classes==0, model is set in regression task
        """
        super().__init__()

        self.lstm = nn.LSTM(input_size=52, hidden_size=104, num_layers=2)

        # Logistic Regression
        self.clf = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )

        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def forward(self, x):
        out = self.lstm(x)
        out = self.clf(out.view(out.size(0), -1))
        return out,




