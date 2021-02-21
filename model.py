import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, kernel_size, features_dim=512):
        super(Extractor, self).__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc3 = nn.Linear(2880, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc3(x))
        return x
