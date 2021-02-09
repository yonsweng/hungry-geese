import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from kaggle_environments import make
from main import policy, agent


def process_reward(reward, done):
    if done:
        if reward > 0:
            return 1
        else:
            return -1
    else:
        return 0


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)

    # standardization
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    else:
        returns = returns - returns.mean()

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    # entropy_loss = -torch.tensor(policy.saved_entropies).sum()
    loss = policy_loss  # + entropy_loss
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    del policy.saved_entropies[:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REINFORCE')
    parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                        help='discount factor (default: 0.98)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='interval between status logs (default: 100)')
    parser.add_argument('--step-log-interval', type=int, default=1000, metavar='N',
                        help='interval between step logs (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='G',
                        help='learning rate (default: 1e-5)')
    args = parser.parse_args()

    env = make('hungry_geese', debug=False)
    configuration = env.configuration
    print(configuration)

    # settings & constants
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tb = SummaryWriter(comment=args.__str__())
    eps = np.finfo(np.float32).eps.item()

    # training the agent in the first position
    trainer = env.train([None, 'examples/risk_averse.py', 'examples/risk_averse.py', 'examples/risk_averse.py'])
    policy.train()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # for log
    ep_rewards = deque(maxlen=args.log_interval)
    ep_steps = deque(maxlen=args.log_interval)
    actions = deque(maxlen=args.step_log_interval)
    action2int = {'NORTH': 0, 'EAST': 1, 'WEST': 2, 'SOUTH': 3}

    for i_episode in range(1, 50001):
        observation, ep_reward = trainer.reset(), 0

        for step in range(1, 101):
            action = agent(observation, configuration, save=True)
            observation, reward, done, info = trainer.step(action)
            reward = process_reward(reward, done)
            policy.rewards.append(reward)
            ep_reward += reward

            actions.append(action2int[action])

            if done:
                break

        ep_rewards.append(ep_reward)
        ep_steps.append(step)
        finish_episode()

        if i_episode % args.log_interval == 0:
            avg_reward = sum(ep_rewards) / len(ep_rewards)
            avg_step = sum(ep_steps) / len(ep_steps)
            print(f'Episode {i_episode}\t'
                  f'Average reward: {avg_reward:.2f}\t'
                  f'Average steps: {avg_step:.2f}\t')
            tb.add_scalar('reward', avg_reward, i_episode)
            tb.add_scalar('step', avg_step, i_episode)
            tb.add_histogram('actions', np.array(actions), i_episode)
            torch.save(policy.state_dict(), 'models/policy.pt')

    tb.close()
