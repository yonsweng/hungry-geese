import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from kaggle_environments import make
from main import ACTION_NAMES, device, policy, preprocess, opposite, agent, Policy
from value_model import Value


def process_reward(obs, reward, done):
    if done:
        if reward > 0:
            return 10
        else:
            return -10
    else:
        # if kill, get 10
        curr_alive = 0
        for goose in obs['geese']:
            if len(goose) > 0:
                curr_alive += 1
        killed = process_reward.prev_alive - curr_alive
        process_reward.prev_alive = curr_alive

        curr_len = reward - 99
        len_diff = curr_len - process_reward.prev_len
        process_reward.prev_len = curr_len
        return killed * 10 + len_diff * 3 + 1


def finish_episode():
    epi_len = len(policy.rewards)
    sum_reward = 0.
    td_targets = [0.] * epi_len
    gamma_td = args.gamma ** args.td
    for i in range(epi_len - 1, -1, -1):
        tail = i + args.td + 1
        if tail < epi_len:
            sum_reward -= gamma_td * policy.rewards[tail]
        sum_reward = args.gamma * sum_reward + policy.rewards[i]
        td_targets[i] = sum_reward + \
            (gamma_td * args.gamma * value.values[tail] if tail < epi_len else torch.tensor([[0.]], device=device))

    log_probs = torch.cat(policy.saved_log_probs)
    td_targets = torch.cat(td_targets).squeeze()
    values = torch.cat(value.values).squeeze()
    policy_loss = (-log_probs * (td_targets - values).detach()).sum()
    value_loss = F.mse_loss(td_targets.detach(), values, reduction='sum')

    optimizer.zero_grad()
    value_optim.zero_grad()
    policy_loss.backward()
    value_loss.backward()
    optimizer.step()
    value_optim.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]
    del policy.saved_entropies[:]
    del value.values[:]


def opponent(observation, configuration):
    index = observation['index'] - 1
    observation = preprocess(observation).to(device)
    logits = oppo_policy[index](observation)
    if last_a[index] != -1:
        # remove illigal move
        illigal_move = opposite(last_a[index])
        mask = torch.arange(4)
        mask = mask[mask != illigal_move].to(device)
        probs = torch.zeros_like(logits, device=device)
        probs[:, mask] = F.softmax(logits[:, mask], 1)
    else:
        probs = F.softmax(logits, 1)
    m = Categorical(probs)
    action = m.sample()
    last_a[index] = action.item()
    action = ACTION_NAMES[last_a[index]]
    return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REINFORCE')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='interval between status logs (default: 100)')
    parser.add_argument('--step-log-interval', type=int, default=1000, metavar='N',
                        help='interval between step logs (default: 1000)')
    parser.add_argument('--change-interval', type=int, default=1000, metavar='N',
                        help='interval between changing opponent policies (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--value-lr', type=float, default=1e-4, metavar='G',
                        help='learning rate for value (default: 1e-4)')
    parser.add_argument('--td', type=int, default=0, metavar='N',
                        help='td(n) (default: 0)')
    parser.add_argument('--prev-pi', type=int, default=10, metavar='N',
                        help='number of previous policies to save (default: 10)')
    args = parser.parse_args()

    env = make('hungry_geese', debug=False)
    configuration = env.configuration
    print(configuration)

    # settings & constants
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tag = ','.join([f'{arg}={getattr(args, arg)}' for arg in vars(args)])
    tb = SummaryWriter(comment=tag)
    eps = np.finfo(np.float32).eps.item()

    # training the agent in the first position
    trainer = env.train([None, 'examples/risk_averse.py', 'examples/risk_averse.py', 'random'])
    policy.train()
    oppo_policy = [Policy().to(device) for _ in range(3)]
    value = Value().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    value_optim = optim.Adam(value.parameters(), lr=args.value_lr)
    last_a = [-1, -1, -1]

    # for log
    ep_rewards = deque(maxlen=args.log_interval)
    ep_steps = deque(maxlen=args.log_interval)
    entropies = deque(maxlen=args.step_log_interval)
    actions = deque(maxlen=args.step_log_interval)
    action2int = {'NORTH': 0, 'EAST': 1, 'WEST': 2, 'SOUTH': 3}

    # for self-play
    prev_policies = deque(maxlen=args.prev_pi)
    oppo_index = 0

    for i_episode in range(1, 100001):
        observation, ep_reward = trainer.reset(), 0
        process_reward.prev_len = 1
        process_reward.prev_alive = 4

        for step in range(1, 101):
            action = agent(observation, configuration, save=True)
            v = value(preprocess(observation).to(device))
            observation, reward, done, info = trainer.step(action)
            reward = reward if step != 1 else reward - 100
            reward = process_reward(observation, reward, done)
            policy.rewards.append(reward)
            value.values.append(v)
            ep_reward += reward

            actions.append(action2int[action])

            if done:
                break

        ep_rewards.append(ep_reward)
        ep_steps.append(step)
        for entropy in torch.cat(policy.saved_entropies).tolist():
            entropies.append(entropy)
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
            tb.add_histogram('entropy', np.array(entropies), i_episode)
            torch.save(policy.state_dict(), f'models/policy_{tag}.pt')

        if i_episode % args.change_interval == 0:
            prev_policies.append(policy.state_dict())
            if len(prev_policies) == args.prev_pi:
                random_index = random.randint(0, len(prev_policies) - 1)
                trainer = env.train([None, opponent, opponent, opponent])
                oppo_policy[oppo_index].load_state_dict(prev_policies[random_index])
                oppo_index = (oppo_index + 1) % 3

    tb.close()
