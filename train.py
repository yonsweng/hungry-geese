import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from copy import deepcopy
from kaggle_environments import make
from main import Policy, ACTION_NAMES, device, policy, preprocess, agent


def process_reward(obs, reward, done):
    '''
    winning bonus:
        0.5 if kill
        -1. if killed
    length bonus:
        .1 per length increase
    '''
    killed, len_diff = 0, 0

    if obs.step == 1:
        reward -= 101

    if reward <= 0:  # if lose
        killed = -2
    else:
        # calc killed
        curr_alive = 0
        for goose in obs.geese[1:]:
            if len(goose) > 0:
                curr_alive += 1

        killed = process_reward.prev_alive - curr_alive
        process_reward.prev_alive = curr_alive

        if curr_alive > 0 and done:  # if step 199
            killed = -2

        len_diff = reward - 100

    return killed * 0.5 + len_diff * 0.1


def finish_episode():
    epi_len = len(policy.saved_rewards)
    sum_reward = 0.
    td_targets = [0.] * epi_len
    gamma_td = args.gamma ** args.td
    for i in range(epi_len - 1, -1, -1):
        tail = i + args.td + 1
        if tail < epi_len:
            sum_reward -= gamma_td * policy.saved_rewards[tail]
        sum_reward = args.gamma * sum_reward + policy.saved_rewards[i]
        td_targets[i] = sum_reward + \
            (gamma_td * args.gamma * policy.saved_values[tail]
             if tail < epi_len else torch.tensor([[0.]], device=device))

    log_probs = torch.cat(policy.saved_log_probs)
    td_targets = torch.cat(td_targets).squeeze()
    values = torch.cat(policy.saved_values).squeeze()
    policy_loss = (-log_probs * (td_targets - values).detach()).sum()
    if finish_episode.entropy_coef > 0:
        # spread probs
        policy_loss += -sum(policy.saved_entropies).squeeze() \
            * finish_episode.entropy_coef
        finish_episode.entropy_coef -= finish_episode.entropy_coef_reduce
    value_loss = F.smooth_l1_loss(values, td_targets.detach(), reduction='sum')
    loss = policy_loss + args.value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del policy.saved_rewards[:]
    del policy.saved_log_probs[:]
    del policy.saved_entropies[:]
    del policy.saved_values[:]
    del policy.saved_logits[:]


def opponent(observation, configuration):
    index = observation.index - 1
    observation = preprocess(observation).to(device)
    logits, _ = oppo_policy[index](observation)
    probs = F.softmax(logits, 1)
    m = Categorical(probs)
    if last_a[index] != -1:
        # remove illigal move
        illigal_move = 3 - last_a[index]  # opposite
        probs2 = probs.clone()
        probs2[:, illigal_move] = 0
        probs2 /= probs2.sum()
        action = Categorical(probs2).sample()
    else:
        action = m.sample()
    last_a[index] = action.item()
    action = ACTION_NAMES[last_a[index]]
    return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A2C')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                        help='discount factor (default: 0.9)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='interval between status logs (default: 500)')
    parser.add_argument('--step-log-interval', type=int, default=1000,
                        metavar='N',
                        help='interval between step logs (default: 1000)')
    parser.add_argument('--change-interval', type=int, default=1000,
                        metavar='N',
                        help='interval btw changing opponent (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='G',
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--value-coef', type=float, default=1, metavar='G',
                        help='coefficient for value loss (default: 1)')
    parser.add_argument('--l2', type=float, default=0, metavar='G',
                        help='l2 regularization (default: 0)')
    parser.add_argument('--td', type=int, default=0, metavar='N',
                        help='td(n) (default: 0)')
    parser.add_argument('--prev-pi', type=int, default=15, metavar='N',
                        help='# of previous policies to save (default: 15)')
    parser.add_argument('--start-self', type=int, default=1, metavar='N',
                        help='episode # to start self-play (default: 1)')
    # for train resume
    parser.add_argument('--entropy-steps', type=int, default=0,
                        metavar='N', help='steps for entropy loss (0: off)')
    parser.add_argument('--load', type=str, metavar='S', default='',
                        help='loading model name')
    args = parser.parse_args()

    env = make('hungry_geese', debug=False)
    configuration = env.configuration
    print(configuration)

    # settings & constants
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    eps = np.finfo(np.float32).eps.item()
    if args.entropy_steps > 0:
        finish_episode.entropy_coef = 1
        finish_episode.entropy_coef_reduce = 1 / args.entropy_steps
    else:
        finish_episode.entropy_coef = 0

    # training the agent in the first position
    trainer = env.train([None,
                         'examples/simple_bfs.py',
                         'examples/risk_averse.py',
                         'examples/mighty_boiler_goose.py'])
    policy.train()
    oppo_policy = [Policy().to(device).eval() for _ in range(3)]
    optimizer = optim.RMSprop(policy.parameters(), lr=args.lr,
                              weight_decay=args.l2)
    last_a = [-1, -1, -1]

    # load model
    if args.load != '':
        policy.load_state_dict(torch.load(args.load, device))

    # for log
    tag = ','.join([f'{arg}={getattr(args, arg)}'
                    for arg in vars(args) if arg != 'load'])
    tb = SummaryWriter(comment=tag)
    running_reward, running_steps = 0, 0
    entropies = deque(maxlen=args.step_log_interval)
    values = deque(maxlen=args.step_log_interval)
    actions = deque(maxlen=args.step_log_interval)
    logits = deque(maxlen=args.step_log_interval)
    action2int = {'NORTH': 0, 'EAST': 1, 'WEST': 2, 'SOUTH': 3}

    # for self-play
    prev_policies = deque(maxlen=args.prev_pi)
    prev_policies.append(policy.state_dict())
    oppo_index = 0

    for i_episode in range(1, 200001):
        # self-play start
        if i_episode == args.start_self:
            for i in range(3):
                random_index = random.randint(0, len(prev_policies) - 1)
                oppo_policy[i].load_state_dict(prev_policies[random_index])
            trainer = env.train([None, opponent, opponent, opponent])

        observation, ep_reward = trainer.reset(), 0
        process_reward.prev_alive = 3

        for step in range(1, 201):
            action = agent(observation, configuration, train=True)
            observation, reward, done, info = trainer.step(action)
            reward = process_reward(observation, reward, done)
            policy.saved_rewards.append(reward)
            ep_reward += reward
            actions.append(action2int[action])
            if done:
                break

        running_reward = running_reward * 0.99 + ep_reward * 0.01
        running_steps = running_steps * 0.99 + step * 0.01
        entropies += deque(torch.cat(policy.saved_entropies).tolist())
        values += deque(torch.cat(policy.saved_values).tolist())
        logits += deque(torch.cat(policy.saved_logits).reshape(-1))

        finish_episode()

        if i_episode % args.log_interval == 0:
            print(f'Episode {i_episode}\t'
                  f'Average reward: {running_reward:.2f}\t'
                  f'Average steps: {running_steps:.2f}\t')
            tb.add_scalar('reward', running_reward, i_episode)
            tb.add_scalar('steps', running_steps, i_episode)
            tb.add_histogram('actions', np.array(actions), i_episode,
                             bins=7)
            tb.add_histogram('entropies', np.array(entropies), i_episode)
            tb.add_histogram('values', np.array(values), i_episode)
            tb.add_histogram('logits', np.array(logits), i_episode)
            torch.save(policy.state_dict(), f'models/policy_{tag}.pt')

        if i_episode > args.start_self \
                and i_episode % args.change_interval == 0:
            prev_policies.append(deepcopy(policy.state_dict()))
            random_index = random.randint(0, len(prev_policies) - 1)
            oppo_policy[oppo_index].load_state_dict(
                prev_policies[random_index])
            oppo_index = (oppo_index + 1) % 3

    tb.close()
