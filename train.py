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
from main import Policy, ACTION_NAMES, device, policy, preprocess, agent
from value_model import Value


def process_reward(obs, reward, done):
    if obs['step'] == 1:
        reward -= 101

    if done:
        if reward > 0:
            return 10
        else:
            return -10

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
    return killed * 10 + len_diff * 2 + 1


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
    value_loss = F.smooth_l1_loss(td_targets.detach(), values, reduction='sum')

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
        illigal_move = 3 - last_a[index]  # opposite
        mask = [a for a in range(4) if a != illigal_move]
        mask = torch.tensor(mask, device=device)
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
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='interval between status logs (default: 200)')
    parser.add_argument('--step-log-interval', type=int, default=1000, metavar='N',
                        help='interval between step logs (default: 1000)')
    parser.add_argument('--change-interval', type=int, default=2000, metavar='N',
                        help='interval between changing opponent policies (default: 2000)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='G',
                        help='learning rate (default: 3e-5)')
    parser.add_argument('--value-lr', type=float, default=3e-5, metavar='G',
                        help='learning rate for value (default: 3e-5)')
    parser.add_argument('--td', type=int, default=0, metavar='N',
                        help='td(n) (default: 0)')
    parser.add_argument('--prev-pi', type=int, default=10, metavar='N',
                        help='number of previous policies to save (default: 20)')
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
    running_reward, running_steps = 0, 0
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
            reward = process_reward(observation, reward, done)
            policy.rewards.append(reward)
            value.values.append(v)
            ep_reward += reward
            actions.append(action2int[action])
            if done:
                break

        running_reward = running_reward * 0.95 + ep_reward * 0.05
        running_steps = running_steps * 0.95 + step * 0.05
        entropies += deque(torch.cat(policy.saved_entropies).tolist())

        finish_episode()

        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\t'
                  f'Average reward: {running_reward:.2f}\t'
                  f'Average steps: {running_steps:.2f}\t')
            tb.add_scalar('reward', running_reward, i_episode)
            tb.add_scalar('steps', running_steps, i_episode)
            tb.add_histogram('actions', np.array(actions), i_episode)
            tb.add_histogram('entropies', np.array(entropies), i_episode)
            torch.save(policy.state_dict(), f'models/policy_{tag}.pt')

        if i_episode % args.change_interval == 0:
            prev_policies.append(policy.state_dict())
            if len(prev_policies) == args.prev_pi - 1:
                for i in range(1, 3):
                    random_index = random.randint(0, len(prev_policies) - 1)
                    oppo_policy[i].load_state_dict(prev_policies[random_index])
            elif len(prev_policies) == args.prev_pi:
                random_index = random.randint(0, len(prev_policies) - 1)
                trainer = env.train([None, opponent, opponent, opponent])
                oppo_policy[oppo_index].load_state_dict(prev_policies[random_index])
                oppo_index = (oppo_index + 1) % 3

    tb.close()
