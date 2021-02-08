import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from kaggle_environments import make
from agent import ACTION_NAMES, device, Policy, policy, preprocess
# from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration

parser = argparse.ArgumentParser(description='REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default:100)')
parser.add_argument('--update-interval', type=int, default=2000, metavar='N',
                    help='interval between opponent update (default:2000)')
parser.add_argument('--lr', type=float, default=1e-6, metavar='G',
                    help='learning rate (default: 1e-6)')
args = parser.parse_args()

env = make('hungry_geese', debug=False)
print(env.configuration)

# Settings & constants
np.random.seed(args.seed)
torch.manual_seed(args.seed)
eps = np.finfo(np.float32).eps.item()

# Load model
policy.train()
oppo = Policy()
oppo.load_state_dict(policy.state_dict())
oppo.to(device)
oppo.eval()

last_a = [torch.tensor([[0., 0., 0., 0.]], device=device)] * 4


def opponent(observation, configuration):
    global oppo, last_a
    # swap
    index = observation['index']
    tmp = observation['geese'][0]
    observation['geese'][0] = observation['geese'][index]
    observation['geese'][index] = tmp
    observation = preprocess(observation).to(device)
    probs = oppo(observation, last_a[index])
    action = Categorical(probs).sample()
    action_name = ACTION_NAMES[action.item()]
    last_a[index] = F.one_hot(action, 4).float().to(device)
    return action_name


# Training agent in first position against agent in second position.
trainer = env.train([None, 'random', 'random', 'random'])
optimizer = optim.Adam(policy.parameters(), lr=args.lr)

prev_len = 1
random_prob = 0.8


def select_action(observation):
    global last_a
    index = observation['index']
    observation = preprocess(observation)
    observation = observation.to(device)
    probs = policy(observation, last_a[index])
    m = Categorical(probs)

    action = m.sample()
    # if random.random() < random_prob \
    # else torch.tensor([random.randint(0, 3)], device=device)

    policy.saved_log_probs.append(m.log_prob(action))
    action_name = ACTION_NAMES[action.item()]
    last_a[index] = F.one_hot(action, 4).float().to(device)
    return action_name


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def process_reward(observation, reward, done):
    global prev_len
    step_bonus = 0
    length_bonus = 0
    winning_bonus = 0

    if observation['step'] == 1:
        reward -= 101

    if done:
        if reward > 0:
            curr_len = reward - 99
            length_bonus = curr_len - prev_len
            prev_len = curr_len

            winning_bonus = 1
        else:
            winning_bonus = -1
    else:
        curr_len = reward - 99
        length_bonus = curr_len - prev_len
        prev_len = curr_len

    reward = step_bonus + length_bonus * 1 + winning_bonus * 10
    return reward


tb = SummaryWriter()
best_score = 0
ep_rewards = deque(maxlen=args.log_interval)
ep_steps = deque(maxlen=args.log_interval)

for i_episode in range(1, 500000):
    observation, ep_reward = trainer.reset(), 0
    prev_len = 1

    for step in range(200):
        action = select_action(observation)
        observation, reward, done, info = trainer.step(action)
        reward = process_reward(observation, reward, done)
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break

    ep_rewards.append(ep_reward)
    ep_steps.append(step)
    finish_episode()

    if i_episode % args.log_interval == 0:
        avg_reward = sum(ep_rewards) / len(ep_rewards)
        avg_step = sum(ep_steps) / len(ep_steps)
        print('Episode {}\tAverage reward: {:.2f}'.format(
              i_episode, avg_reward))
        tb.add_scalar('reward', avg_reward, i_episode)
        tb.add_scalar('step', avg_step, i_episode)
        if avg_reward > best_score:
            best_score = avg_reward
        torch.save(policy.state_dict(), './models/policy.pt')

    if i_episode % args.update_interval == 0:
        # Opponent update
        oppo.load_state_dict(policy.state_dict())
        random_prob += 0.01

tb.close()
