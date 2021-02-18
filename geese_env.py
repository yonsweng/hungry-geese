# Initial template from: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
# Modified from https://www.kaggle.com/victordelafuente/dqn-goose-with-stable-baselines-wip
from os import listdir
from os.path import isfile, join, getctime
import random
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class HungryGeeseEnv(gym.Env):
    def __init__(self, save_path, self_play_start, debug=False):
        super(HungryGeeseEnv, self).__init__()

        self.save_path = save_path
        self.debug = debug
        self.actions = [action for action in Action]
        self.action_offset = 1
        self.env = make("hungry_geese", debug=self.debug)
        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns
        self.hunger_rate = self.env.configuration.hunger_rate
        self.min_food = self.env.configuration.min_food

        self.obs_prev = None
        self.past_models = [None, None, None]
        self.last_actions = [-1, -1, -1]
        self.random_eps = 0.1
        self.change_index = 0

        self.self_play = False
        self.trainer = self.env.train([
            None,
            'examples/simple_bfs.py',
            'examples/mighty_boiler_goose.py',
            'examples/risk_averse.py'
        ])

        if self_play_start == 0:
            self.init_self_play()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(18, self.rows, self.columns),
            dtype=np.float32
        )

    def init_self_play(self):
        self.self_play = True
        self.trainer = self.env.train(
            [None, self.opponent, self.opponent, self.opponent]
        )

    def change_model(self):
        path = self.save_path
        try:
            files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
            files = sorted(files, key=getctime, reverse=True)
            model_name = files[random.randrange(min(len(files), 10))]
            self.past_models[self.change_index] = PPO.load(model_name)
            self.change_index = (self.change_index + 1) % len(self.past_models)
        except Exception as e:
            print(e)

    def opponent(self, obs, conf):
        model_index = obs.index - 1
        obs_backup = obs
        if random.random() < self.random_eps:
            illegal_action = (self.last_actions[model_index] + 2) % 4 \
                if self.last_actions[model_index] != -1 else -1
            actions = [a for a in range(4) if a != illegal_action]
            action = actions[random.randrange(len(actions))]
        else:
            obs = self.process_obs(obs)
            action, _ = self.past_models[model_index].predict(obs)
        self.last_actions[model_index] = action
        action += self.action_offset
        if model_index == len(self.past_models) - 1:
            self.obs_prev = obs_backup
        return Action(action).name

    # Modified from https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning
    def process_obs(self, obs):
        b = np.zeros((18, 7 * 11), dtype=np.float32)
        b[-1] = 1  # empty cells

        for p, pos_list in enumerate(obs['geese']):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - obs['index']) % 4, pos] = 1
                b[-1, pos] = 0
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - obs['index']) % 4, pos] = 1
                b[-1, pos] = 0
            # whole position
            for pos in pos_list:
                b[8 + (p - obs['index']) % 4, pos] = 1
                b[-1, pos] = 0

        # previous head position
        if self.obs_prev is not None:
            for p, pos_list in enumerate(self.obs_prev['geese']):
                for pos in pos_list[:1]:
                    b[12 + (p - obs['index']) % 4, pos] = 1

        # food
        for pos in obs['food']:
            b[16, pos] = 1
            b[-1, pos] = 0

        return b.reshape(-1, 7, 11)

    def process_reward(self, obs, done):
        '''
        reward:
            +1   if 1st
            -1/3 if 2nd
            -2/3 if 3rd
            -1   if 4th
        '''
        if done:
            if len(obs.geese[0]) == 0:  # if I'm dead
                rank = 1
                for i, goose in enumerate(obs.geese[1:], 1):
                    if len(goose) > 0:
                        rank += 1
                    else:  # if goose i is dead
                        if self.obs_prev is not None:
                            if len(self.obs_prev.geese[i]) \
                                    > len(self.obs_prev.geese[0]):
                                rank += 1
                            elif len(self.obs_prev.geese[i]) \
                                    == len(self.obs_prev.geese[0]):
                                rank += 0.5
                        else:
                            rank += 0.5
            else:  # if I'm alive
                rank = 1
                for i, goose in enumerate(obs.geese[1:], 1):
                    if len(goose) > len(obs.geese[0]):
                        rank += 1
                    elif len(goose) == len(obs.geese[0]):
                        rank += 0.5
            if rank == 1:
                return 1.
            else:
                return (1 - rank) / 3
        return 0.

    def step(self, action):
        action += self.action_offset
        obs, reward, done, info = self.trainer.step(Action(action).name)
        reward = self.process_reward(obs, done)
        obs = self.process_obs(obs)
        return obs, reward, done, info

    def reset(self):
        self.obs_prev = None
        obs = self.trainer.reset()
        obs = self.process_obs(obs)
        return obs

    def render(self, **kwargs):
        self.env.render(**kwargs)
