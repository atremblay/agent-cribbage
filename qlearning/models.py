import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, num_in, num_out, layers):
        super(MLP, self).__init__()

        assert isinstance(layers, list)

        # Constructs an n-layer architecture.
        arch = []
        for i in range(len(layers)):
            # First layer of network.
            if i == 0:
                arch.append(nn.Linear(num_in, layers[i]))

            # Final layer of network.
            elif i == len(layers)-1:
                arch.append(nn.Linear(layers[i], num_out))

            # Intermediate layers of network.
            else:
                arch.append(nn.Linear(layers[i-1], layers[i]))

            # Activation & BatchNorm for all layers except the last.
            if i != len(layers)-1:
                arch.extend([nn.ReLU(), nn.BatchNorm1d(layers[i])])

        self.mlp = nn.Sequential(*arch)
                
    def forward(self, X):
        return self.mlp(X)
