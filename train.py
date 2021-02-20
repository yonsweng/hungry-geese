# import os
import argparse
import torch.nn as nn
from torch.optim import Adam, RMSprop, SGD
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy
from geese_env import HungryGeeseEnv
# from model import CustomActorCriticPolicy
from callback import CustomCallback

parser = argparse.ArgumentParser(description='Stable-Baselines3 PPO')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--save_path', default='models0', type=str)
parser.add_argument('--n_envs', default=4, type=int)
parser.add_argument('--self_play_start', default=0, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--optim', default='rmsprop', type=str)
parser.add_argument('--ent_coef', default=0.02, type=float)
parser.add_argument('--vf_coef', default=0.2, type=float)
parser.add_argument('--hidden_units', default=512, type=int)
args = parser.parse_args()

env_kwargs = dict(
    save_path=args.save_path,
    self_play_start=args.self_play_start
)
env = make_vec_env(HungryGeeseEnv, n_envs=args.n_envs,
                   env_kwargs=env_kwargs)

if args.load_path != '':
    # import glob
    # list_of_files = glob.glob(os.path.join(args.load_path, '*.zip'))
    # file = max(list_of_files, key=os.path.getctime)
    file = args.load_path
    print('Loading', file)
    model = PPO.load(file, env)
else:
    h = args.hidden_units
    net_arch = [h, h, dict(pi=[h, h], vf=[h, h])]
    if args.optim.lower() == 'sgd':
        optimizer_class = SGD
        optimizer_kwargs = dict(momentum=0.9)
    elif args.optim.lower() == 'rmsprop':
        optimizer_class = RMSprop
        optimizer_kwargs = dict()
    else:
        optimizer_class = Adam
        optimizer_kwargs = dict()
    policy_kwargs = dict(net_arch=net_arch,
                         activation_fn=nn.ReLU,
                         optimizer_class=optimizer_class,
                         optimizer_kwargs=optimizer_kwargs)

    model = PPO(MlpPolicy, env, verbose=0,
                tensorboard_log='runs',
                learning_rate=args.lr,
                clip_range=0.2,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef)

callback = CustomCallback(
    save_freq=10000, save_path=args.save_path, name_prefix='model',
    self_play_start=args.self_play_start
)  # save_freq should be divided by n_envs

model.learn(10000000, callback=callback, reset_num_timesteps=True)
