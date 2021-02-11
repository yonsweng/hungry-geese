import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    """
    Input: observation (1, 3, 77)
    Output: policy (1, 4), value (1, 1)
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.linear0 = nn.Linear(3 * 77, 4096)
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 4)
        self.vlinear = nn.Linear(1024, 1)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.saved_rewards = []
        self.saved_values = []

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.linear0(x))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        p = self.linear3(x)
        v = self.vlinear(x)
        return p, v


# global constants & variables
ACTION_NAMES = ['NORTH', 'EAST', 'WEST', 'SOUTH']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = Policy().to(device).eval()
last_a = -1
eps = 0.1

if __name__ != 'main':
    # load policy
    policy.load_state_dict(torch.load('/kaggle_simulations/agent/policy.pt',
                                      map_location=device))


def preprocess(observation):
    x = torch.zeros(1, 3, 77)
    for food in observation.food:
        x[0, 0, food] = 1.
    for i, goose in enumerate(observation.geese):
        if len(goose) > 0:
            i = 1 if i == observation.index else 2
            x[0, i, goose[0]] = 1.
            for part in goose[1:]:
                x[0, i, part] = -1.
    return x


def agent(observation, configuration, train=False):
    global ACTION_NAMES, device, policy, last_a, eps
    observation = preprocess(observation).to(device)
    logits, value = policy(observation)
    if last_a != -1:
        # remove illigal move
        illigal_move = 3 - last_a  # opposite
        mask = [a for a in range(4) if a != illigal_move]
        mask = torch.tensor(mask, device=device)
        probs = torch.zeros_like(logits, device=device)
        probs[:, mask] = F.softmax(logits[:, mask], 1)
        m = Categorical(probs)
        if train:
            eps *= 0.99999  # reduce random probability
            action = m.sample()  # if random.random() >= eps \
            # else mask[random.randint(0, 2)].unsqueeze(0)
        else:
            action = m.sample()
    else:
        probs = F.softmax(logits, 1)
        m = Categorical(probs)
        action = m.sample()
    if train:
        policy.saved_log_probs.append(m.log_prob(action))
        policy.saved_entropies.append(m.entropy())
        policy.saved_values.append(value)
    last_a = action.item()
    action = ACTION_NAMES[last_a]
    return action
