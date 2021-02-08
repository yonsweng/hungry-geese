import os
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
        self.conv = nn.Sequential(
            nn.Conv1d(3, 8, 1),
            # nn.Dropout(),
            nn.ReLU()
        )
        self.dense = nn.Sequential(
            nn.Linear(8 * 77, 256),
            # nn.Dropout(),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(260, 64)
        self.linear2 = nn.Linear(64, 4)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, a):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = torch.cat((x, a), 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = F.softmax(x, 1)
        return x


def preprocess(observation):
    x = torch.zeros(1, 3, 77)
    for food in observation['food']:
        x[0, 0, food] = 1.
    for i, goose in enumerate(observation['geese'], 1):
        j = i if i < 2 else 2
        if len(goose) > 0:
            x[0, j, goose[0]] = 1.0
        for part in goose[1:]:
            x[0, j, part] = 0.5
    return x


ACTION_NAMES = ['NORTH', 'EAST', 'WEST', 'SOUTH']

policy = Policy()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load policy
# if os.path.exists('../input/hungry-geese-models/policy.pt'):
#     policy.load_state_dict(torch.load('../input/hungry-geese-models/policy.pt', map_location=device))
# elif os.path.exists(os.path.join('/kaggle_simulations/agent/', 'policy.pt')):
#     policy.load_state_dict(torch.load(os.path.join('/kaggle_simulations/agent/', 'policy.pt'), map_location=device))
# else:
policy.load_state_dict(torch.load('models/policy.pt', map_location=device))
# torch.save(policy.state_dict(), 'policy.pt')

policy.to(device)
policy.eval()

last_a = torch.tensor([[0., 0., 0., 0.]], device=device)


def my_agent(observation, configuration):
    global ACTION_NAMES, policy, device, last_a
    observation = preprocess(observation).to(device)
    probs = policy(observation, last_a)
    action = Categorical(probs).sample()
    action_name = ACTION_NAMES[action.item()]
    last_a = F.one_hot(action, 4).float().to(device)
    return action_name