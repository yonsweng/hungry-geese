import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    """
    Input: observation (1, 3, 77)
    Output: value (1, 1)
    """
    def __init__(self):
        super(Value, self).__init__()
        self.linear0 = nn.Linear(3 * 77, 4096)
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 1)
        self.values = []

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
