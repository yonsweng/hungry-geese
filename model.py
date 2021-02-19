import torch.nn as nn
from torch.nn.modules.container import Sequential


class FlattenExtractor(nn.Module):
    """Some Information about FlattenExtractor"""
    def __init__(self):
        super(FlattenExtractor, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        return x


class MlpExtractor(nn.Module):
    """Some Information about MlpExtractor"""
    def __init__(self):
        super(MlpExtractor, self).__init__()
        self.shared_net = Sequential()
        self.policy_net = Sequential(
            nn.Linear(in_features=1386, out_features=64, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.Tanh()
        )
        self.value_net = Sequential(
            nn.Linear(in_features=1386, out_features=64, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=64, out_features=64, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        p = self.policy_net(x)
        v = self.value_net(x)
        return p, v


class ActorCriticPolicy(nn.Module):
    """Some Information about ActorCriticPolicy"""
    def __init__(self):
        super(ActorCriticPolicy, self).__init__()
        self.feature_extractor = FlattenExtractor()
        self.mlp_extractor = MlpExtractor()
        self.action_net = nn.Linear(in_features=64, out_features=4, bias=True)
        self.value_net = nn.Linear(in_features=64, out_features=1, bias=True)

    def forward(self, x):
        x = self.feature_extractor(x)
        p, v = self.mlp_extractor(x)
        p = self.action_net(p)
        v = self.value_net(v)
        return p, v
