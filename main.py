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
        self.conv = nn.Conv1d(3, 32, 1)
        self.linear1 = nn.Linear(32 * 77, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 4)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# global constants & variables
ACTION_NAMES = ['NORTH', 'EAST', 'WEST', 'SOUTH']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = Policy()
last_a = -1

# load policy
# policy.load_state_dict(torch.load('submission/policy.pt', map_location=device))

policy.to(device)
policy.eval()


def preprocess(observation):
    x = torch.zeros(1, 3, 77)
    for food in observation['food']:
        x[0, 0, food] = 1.
    for i, goose in enumerate(observation['geese']):
        if len(goose) > 0:
            i = 1 if i == observation['index'] else 2
            x[0, i, goose[0]] = 1.
            for part in goose[1:]:
                x[0, i, part] = .5
    return x


def opposite(a):
    return 3 - a


def agent(observation, configuration, save=False):
    global ACTION_NAMES, device, policy, last_a
    observation = preprocess(observation).to(device)
    logits = policy(observation)
    if last_a != -1:
        # remove illigal move
        illigal_move = opposite(last_a)
        mask = torch.arange(4)
        mask = mask[mask != illigal_move].to(device)
        probs = torch.zeros_like(logits, device=device)
        probs[:, mask] = F.softmax(logits[:, mask], 1)
    else:
        probs = F.softmax(logits, 1)
    m = Categorical(probs)
    action = m.sample()
    if save:
        policy.saved_log_probs.append(m.log_prob(action))
        policy.saved_entropies.append(m.entropy())
    last_a = action.item()
    action = ACTION_NAMES[last_a]
    return action
