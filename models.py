import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    """
    Input: observation (1, 10 * 77 + 1)
    Output: value (1, 1)
    """
    def __init__(self):
        super(Value, self).__init__()
        self.linear0 = nn.Linear(10 * 77 + 1, 2048)
        self.linear1 = nn.Linear(2048, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        # self.linear3 = nn.Linear(2048, 2048)
        # self.linear4 = nn.Linear(2048, 2048)
        # self.linear5 = nn.Linear(2048, 2048)
        # self.linear6 = nn.Linear(2048, 2048)
        self.linear7 = nn.Linear(2048, 1)
        self.saved_values = []

    def forward(self, x):
        x = F.leaky_relu(self.linear0(x))
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        # x = F.leaky_relu(self.linear3(x))
        # x = F.leaky_relu(self.linear4(x))
        # x = F.leaky_relu(self.linear5(x))
        # x = F.leaky_relu(self.linear6(x))
        x = self.linear7(x)
        return x
