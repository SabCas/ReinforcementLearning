"""
Defines a PyTorch policy network for reinforcement learning.

The PolicyNetwork class is a PyTorch module that represents a neural network policy
for a reinforcement learning agent. It takes in the state space and action space
dimensions, as well as a learning rate, and defines a fully-connected neural network
with three hidden layers. The forward method of the network applies a ReLU activation
to the first two hidden layers, and a softmax activation to the output layer to
produce a probability distribution over the actions.

The network is initialized on the available GPU if one is present, or on the CPU
otherwise.
"""
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x
              

