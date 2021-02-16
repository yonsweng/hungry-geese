import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from geese_env import HungryGeeseEnv
from model import CustomActorCriticPolicy
from callback import CustomCallback

parser = argparse.ArgumentParser(description='Stable-Baselines3 PPO')
parser.add_argument('--load', action='store_true')
args = parser.parse_args()

num_cpu = 4
env = make_vec_env(HungryGeeseEnv, n_envs=num_cpu)

if args.load:
    import glob
    import os
    list_of_files = glob.glob('models/*.zip')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Loading', latest_file)
    model = PPO.load(latest_file, env)
else:
    model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log='runs')

callback = CustomCallback(
    save_freq=8000, save_path='models', name_prefix='model'
)  # save_freq should be divided by n_envs

model.learn(4000000, callback=callback)
