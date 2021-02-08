import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    """
    Input: observation (1, 3, 77)
    Output: policy (1, 4)
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.conv = nn.Conv1d(3, 8, 1)
        self.linear1 = nn.Linear(8 * 77, 256)
        self.linear2 = nn.Linear(260, 64)
        self.linear3 = nn.Linear(64, 4)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, a):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = torch.cat((x, a), 1)
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = F.softmax(x, 1)
        return x


# global constants & variables
ACTION_NAMES = ['NORTH', 'EAST', 'WEST', 'SOUTH']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = Policy()
last_a = torch.tensor([[0., 0., 0., 0.]], device=device)

if __name__ == '__main__':
    # load policy
    policy.load_state_dict(torch.load('submission/policy.pt', map_location=device))

policy.to(device)
policy.eval()


def preprocess(observation):
    x = torch.zeros(1, 3, 77)
    for food in observation['food']:
        x[0, 0, food] = 1.0
    for i, goose in enumerate(observation['geese'], 1):
        i = 2 if i >= 2 else 1
        if len(goose) > 0:
            x[0, i, goose[0]] = 1.0
        for part in goose[1:]:
            x[0, i, part] = 0.5
    return x


def agent(observation, configuration, save=False):
    global ACTION_NAMES, device, policy, last_a
    observation = preprocess(observation).to(device)
    probs = policy(observation, last_a)
    m = Categorical(probs)
    action = m.sample()
    if save:
        policy.saved_log_probs.append(m.log_prob(action))
    last_a = F.one_hot(action, 4).float().to(device)
    action = ACTION_NAMES[action.item()]
    return action
