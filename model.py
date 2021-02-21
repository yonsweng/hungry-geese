import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, kernel_size, features_dim=64*7*11):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        return x
