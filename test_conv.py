
from agent.value_function.conv import Conv
import numpy as np
import torch


ranks = np.zeros((1, 1, 13, 4), dtype=np.float32)
suits = np.zeros((1, 4, 4), dtype=np.float32)
ranks[0, 0, 0, 0] = 1
ranks[0, 0, 0, 1] = 1
ranks[0, 0, 0, 2] = 1
ranks[0, 0, 0, 3] = 1

suits[0, 0, 0] = 1
suits[0, 1, 0] = 1
suits[0, 2, 0] = 1
suits[0, 3, 0] = 1

ranks = torch.tensor(ranks)
suits = torch.tensor(suits)


model = Conv()
model(ranks, suits)
