import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from geese_env import HungryGeeseEnv
from model import CustomActorCriticPolicy
from callback import CustomCallback

parser = argparse.ArgumentParser(description='Stable-Baselines3 PPO')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--save_path', default='models', type=str)
parser.add_argument('--n_envs', default=4, type=int)
args = parser.parse_args()

if args.load_path != '':
    import glob
    import os
    list_of_files = glob.glob(os.path.join(args.load_path, '*.zip'))
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Loading', latest_file)
    model = PPO.load(latest_file)
else:
    env_kwargs = dict(save_path=args.save_path)
    env = make_vec_env(HungryGeeseEnv, n_envs=args.n_envs,
                       env_kwargs=env_kwargs)
    model = PPO(CustomActorCriticPolicy, env, verbose=0,
                tensorboard_log='runs')

callback = CustomCallback(
    save_freq=10000, save_path=args.save_path, name_prefix='model'
)  # save_freq should be divided by n_envs

model.learn(10000000, callback=callback, reset_num_timesteps=True)
