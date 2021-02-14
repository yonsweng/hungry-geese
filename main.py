import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    """
    Input: observation (1, 10 * 77 + 1)
    Output: policy (1, 4)
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.linear0 = nn.Linear(10 * 77 + 1, 2048)
        self.linear1 = nn.Linear(2048, 2048)
        self.linear2 = nn.Linear(2048, 2048)
        # self.linear3 = nn.Linear(2048, 2048)
        # self.linear4 = nn.Linear(2048, 2048)
        # self.linear5 = nn.Linear(2048, 2048)
        # self.linear6 = nn.Linear(2048, 2048)
        self.linear7 = nn.Linear(2048, 4)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.saved_rewards = []
        self.saved_logits = []

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


# global constants & variables
ACTION_NAMES = ['NORTH', 'EAST', 'WEST', 'SOUTH']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = Policy().to(device).eval()
last_a = -1

if __name__ != 'main':
    # load policy
    policy.load_state_dict(torch.load('/kaggle_simulations/agent/policy.pt',
                                      map_location=device))


def preprocess(observation, configuration):
    ''' One-hot encoding
        x[0, 0, :] : emtpy cells
        x[0, 1, :] : my head
        x[0, 2, :] : my body
        x[0, 3, :] : opponent 0 head
        x[0, 4, :] : opponent 0 body
        x[0, 5, :] : opponent 1 head
        x[0, 6, :] : opponent 1 body
        x[0, 7, :] : opponent 2 head
        x[0, 8, :] : opponent 2 body
        x[0, 9, :] : foods
    '''

    # initialize
    x = np.zeros((1, 10, 77), dtype=np.float32)

    # empty cells
    x[0, 0, :] = 1

    # my head and body
    my_goose = observation.geese[observation.index]
    my_head_pos = my_goose[0]
    x[0, 1, my_head_pos] = 1
    x[0, 0, my_head_pos] = 0
    for my_body_pos in my_goose[1:]:
        x[0, 2, my_body_pos] = 1
        x[0, 0, my_body_pos] = 0

    # opponents' heads and bodies
    opponent_index = 0
    for i, goose in enumerate(observation.geese):
        if i != observation.index and len(goose) > 0:
            head_index = 2 * opponent_index + 3
            body_index = 2 * opponent_index + 4
            opponent_index += 1
            x[0, head_index, goose[0]] = 1
            x[0, 0, goose[0]] = 0
            for body_pos in goose[1:]:
                x[0, body_index, body_pos] = 1
                x[0, 0, body_pos] = 0

    # foods
    for food_pos in observation.food:
        x[0, 9, food_pos] = 1
        x[0, 0, food_pos] = 0

    step_norm = np.array([[observation.step % configuration.hunger_rate
                           / configuration.hunger_rate]], dtype=np.float32)
    x = np.concatenate((x.reshape(1, -1), step_norm), axis=1)

    return torch.from_numpy(x).to(device)


def agent(observation, configuration, train=False):
    global ACTION_NAMES, device, policy, last_a
    observation = preprocess(observation, configuration)
    logits = policy(observation)
    m = Categorical(logits=logits)
    if last_a != -1:
        illegal_move = 3 - last_a  # opposite
        probs = F.softmax(logits, 1)
        probs[:, illegal_move] = 0
        probs /= probs.sum()
        action = Categorical(probs).sample()
    else:
        action = m.sample()
    if train:
        policy.saved_log_probs.append(m.log_prob(action))
        policy.saved_entropies.append(m.entropy())
        policy.saved_logits.append(logits)
    last_a = action.item()
    action = ACTION_NAMES[last_a]
    if train:
        return action, observation
    return action
