import torch.nn as nn
import torch


class ValueFunction(nn.Module):
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()

    def update(self, batch):
        pass

    def evaluate(self, x):
        self.eval()
        return self.forward(torch.tensor(x, dtype=torch.float))

    def forward(self, x):
        pass