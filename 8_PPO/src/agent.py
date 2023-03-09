import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            self.layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            self.layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        value = self.critic(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logprob = probs.log_prob(action)
        return action, logprob, probs.entropy(), value
