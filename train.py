import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from kaggle_environments import make
from models import GeeseNet


'''
make_input and agent functions
Copied from
https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning
Modified by
yonsweng (Choi Yeonung)
in Feb 2021
'''
obs_prev = None
action_prev = [None] * 4


def make_input(obs):
    b = np.zeros((17, 7 * 11), dtype=np.float32)

    for p, pos_list in enumerate(obs['geese']):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs['index']) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs['index']) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs['index']) % 4, pos] = 1

    # previous head position
    if obs_prev is not None:
        for p, pos_list in enumerate(obs_prev['geese']):
            for pos in pos_list[:1]:
                b[12 + (p - obs['index']) % 4, pos] = 1

    # food
    for pos in obs['food']:
        b[16, pos] = 1

    return b.reshape(-1, 7, 11)


def agent(obs, _, train=False):
    global obs_prev

    x = make_input(obs)
    if train:
        obs_prev = obs

    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    o = model(xt)

    p = o['policy'].squeeze(0)
    m = Categorical(logits=p)

    # if action_prev[obs.index] is None:
    action = m.sample()
    # else:
    #     # remove illegal action
    #     illegal_action = action_prev[obs.index] + 1 \
    #                      if action_prev[obs.index] % 2 == 0 \
    #                      else action_prev[obs.index] - 1
    #     probs = F.softmax(p, dim=0)
    #     probs[illegal_action] = 0
    #     probs /= probs.sum()
    #     action = Categorical(probs).sample()

    if train:
        model.saved_log_probs.append(m.log_prob(action))
        model.saved_values.append(o['value'].squeeze())

    actions = ['NORTH', 'SOUTH', 'WEST', 'EAST']
    action = action.item()
    action_prev[obs.index] = action
    return actions[action]


def process_reward(obs, done):
    '''
    reward:
        +1   if 1st
        1/3  if 2nd
        -1/3 if 3rd
        -1   if 4th
    '''
    if done:
        if len(obs.geese[0]) == 0:  # if I'm dead
            rank = 1
            for i, goose in enumerate(obs.geese[1:], 1):
                if len(goose) > 0:
                    rank += 1
                else:  # if goose i is dead
                    if len(obs_prev.geese[i]) > len(obs_prev.geese[0]):
                        rank += 1
                    elif len(obs_prev.geese[i]) == len(obs_prev.geese[0]):
                        rank += 0.5
        else:  # if I'm alive
            rank = 1
            for i, goose in enumerate(obs.geese[1:], 1):
                if len(goose) > len(obs.geese[0]):
                    rank += 1
                elif len(goose) == len(obs.geese[0]):
                    rank += 0.5
        return (3 - 2 * (rank - 1)) / 3
    return 0.


def finish_episode():
    epi_len = len(model.saved_rewards)
    td_targets = [0.] * epi_len
    sum_reward = 0.
    gamma_td = args.gamma ** args.td
    for i in range(epi_len - 1, -1, -1):
        tail = i + args.td + 1
        if tail < epi_len:
            sum_reward -= gamma_td * model.saved_rewards[tail]
        sum_reward = args.gamma * sum_reward + model.saved_rewards[i]
        td_targets[i] = sum_reward + \
            (gamma_td * args.gamma * model.saved_values[tail]
             if tail < epi_len else torch.tensor(0., device=device))

    log_probs = torch.stack(model.saved_log_probs)
    td_targets = torch.stack(td_targets)
    values = torch.stack(model.saved_values)

    policy_loss = -(log_probs * (td_targets - values).detach()).sum()
    value_loss = F.smooth_l1_loss(values, td_targets.detach(), reduction='sum')
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del model.saved_log_probs[:]
    del model.saved_rewards[:]
    del model.saved_values[:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='interval between status logs (default: 500)')
    parser.add_argument('--n-epi', type=int, default=500000, metavar='N',
                        help='number of episodes to train (default: 500000)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='G',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--td', type=int, default=0, metavar='N',
                        help='td(n) (default: 0)')
    parser.add_argument('--load', type=str, metavar='S', default='',
                        help='model name to load')
    args = parser.parse_args()

    # environment
    env = make('hungry_geese', debug=False)
    trainer = env.train([None, agent, agent, agent])

    # settings & constants
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create model
    model = GeeseNet()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load weights
    if args.load != '':
        model.load_state_dict(torch.load(args.load, map_location=device))

    model.to(device)

    # for log
    tag = ','.join([f'{arg}={getattr(args, arg)}'
                    for arg in vars(args) if arg != 'load'])
    running_reward, running_steps = 0, 0

    # for tensorboard
    tb = SummaryWriter(comment=tag)

    for i_episode in range(1, args.n_epi):
        obs, ep_reward = trainer.reset(), 0

        for step in range(1, 200):
            action = agent(obs, None, train=True)
            obs, reward, done, info = trainer.step(action)
            reward = process_reward(obs, done)
            model.saved_rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = running_reward * 0.99 + ep_reward * 0.01
        running_steps = running_steps * 0.99 + step * 0.01

        finish_episode()

        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode} \t'
                  f'Average reward: {running_reward:.2f} \t'
                  f'Average steps: {running_steps:.2f} \t')
            tb.add_scalar('reward', running_reward, i_episode)
            tb.add_scalar('steps', running_steps, i_episode)
            torch.save(model.state_dict(), f'models/{tag}.pt')

    tb.close()
