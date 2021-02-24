import argparse
import torch.nn as nn
from torch.optim import Adam, RMSprop
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy
from geese_env import HungryGeeseEnv
from model import FeaturesExtractor
from callback import CustomCallback
# from scheduler import ExponentialWarmUpSceduler

parser = argparse.ArgumentParser(description='Stable-Baselines3 PPO')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--save_path', default='models0', type=str)
parser.add_argument('--n_envs', default=4, type=int)
parser.add_argument('--self_play_start', default=2000000, type=int)
# parser.add_argument('--init_lr', default=1e-6, type=float)
parser.add_argument('--lr', default=1e-4, type=float)
# parser.add_argument('--schedule_steps', default=500000, type=int)
parser.add_argument('--optim', default='rmsprop', type=str)
parser.add_argument('--alpha', default=0.99, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--ent_coef', default=0.01, type=float)
parser.add_argument('--vf_coef', default=0.5, type=float)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--n_steps', default=2048, type=int)
args = parser.parse_args()
print(args)

env_kwargs = dict(
    save_path=args.save_path,
    self_play_start=args.self_play_start
)
env = make_vec_env(HungryGeeseEnv, n_envs=args.n_envs,
                   env_kwargs=env_kwargs)

if args.load_path != '':
    model = PPO.load(args.load_path, env)
else:
    if args.optim.lower() == 'rmsprop' or args.optim.lower() == 'rms':
        optimizer_class = RMSprop
        optimizer_kwargs = dict(
            alpha=args.alpha,
            weight_decay=args.weight_decay
        )
    else:
        optimizer_class = Adam
        optimizer_kwargs = dict(
            weight_decay=args.weight_decay
        )

    net_arch = [512, 256, dict(pi=[128, 64], vf=[128, 64])]

    policy_kwargs = dict(
        net_arch=net_arch,
        features_extractor_class=FeaturesExtractor,
        features_extractor_kwargs=None,
        activation_fn=nn.ReLU,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs
        # lr_schedule=ExponentialWarmUpSceduler(args.init_lr, args.lr, args.schedule_steps)
    )

    model = PPO(MlpPolicy, env, verbose=0,
                tensorboard_log='runs',
                learning_rate=args.lr,
                clip_range=0.2,
                gamma=args.gamma,
                policy_kwargs=policy_kwargs,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                n_steps=args.n_steps)

# save_freq should be divided by n_envs
callback = CustomCallback(
    save_freq=50000, save_path=args.save_path, name_prefix='model',
    self_play_start=args.self_play_start, verbose=1
)

model.learn(10000000, callback=callback)
