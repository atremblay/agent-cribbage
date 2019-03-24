import torch.nn as nn
import torch

class ValueFunction:
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def update(self, batch):
        pass

    def evaluate(self, x, actions):
        return self.forward(torch.tensor([x.state, actions.state], dtype=torch.float).flatten())

    def forward(self, x):
        pass