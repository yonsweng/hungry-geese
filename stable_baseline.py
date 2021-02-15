from geese_env import HungryGeeseEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from callback import CustomCallback

env = make_vec_env(HungryGeeseEnv, n_envs=16)

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log='ppo_logs')
callback = CustomCallback(
    save_freq=16000, save_path='ppo_models', name_prefix='model'
)  # save_freq should be divided by n_envs
model.learn(8000000, callback=callback)
