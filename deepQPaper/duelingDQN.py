"""
Defines a Dueling Deep Q-Network (Dueling DQN) model for reinforcement learning tasks.

The Dueling DQN model separates the estimation of state values and advantages.
"""

import torch
import torch.nn as nn
import numpy as np

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # Advantage stream
        self.advantage_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Value stream
        self.value_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size(0), -1)
        
        advantage = self.advantage_fc(conv_out)
        value = self.value_fc(conv_out)
        
        # Combine value and advantage streams to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values