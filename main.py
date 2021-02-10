import random
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
        self.linear0 = nn.Linear(3 * 77, 4096)
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 4)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# global constants & variables
ACTION_NAMES = ['NORTH', 'EAST', 'WEST', 'SOUTH']
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
policy = Policy()
eps = 0.05
last_a = -1

# load policy
# policy.load_state_dict(torch.load('/kaggle_simulations/agent/policy.pt', map_location=device))

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
    global ACTION_NAMES, device, policy, last_a, eps
    observation = preprocess(observation).to(device)
    logits = policy(observation)
    if last_a != -1:
        # remove illigal move
        illigal_move = opposite(last_a)
        mask = [a for a in range(4) if a != illigal_move]
        mask = torch.tensor(mask, device=device)
        probs = torch.zeros_like(logits, device=device)
        probs[:, mask] = F.softmax(logits[:, mask], 1)
        m = Categorical(probs)
        action = m.sample() if random.random() >= eps \
            else mask[random.randint(0, 2)].unsqueeze(0)
    else:
        probs = F.softmax(logits, 1)
        m = Categorical(probs)
        action = m.sample()
    if save:
        policy.saved_log_probs.append(m.log_prob(action))
        policy.saved_entropies.append(m.entropy())
    last_a = action.item()
    action = ACTION_NAMES[last_a]
    eps *= 0.9999  # reduce random probability
    return action
