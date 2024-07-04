"""
The ActorNetwork class is a PyTorch module that represents the actor network in a Deep Deterministic Policy Gradient (DDPG) reinforcement learning algorithm.

The actor network is responsible for mapping the current state of the environment to an action that the agent should take. It consists of a series of fully connected layers with batch normalization and ReLU activation functions, followed by a final layer that outputs the action.

The constructor of the ActorNetwork class takes the following parameters:
- state_space: the size of the state space
- action_space: the size of the action space
- lr: the learning rate for the Adam optimizer used to train the network

The forward method of the class takes a state as input and returns the corresponding action.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(ActorNetwork, self).__init__()
        self.lr = lr
        self.fc1 = nn.Linear(state_space, 400)
        self.fc2 = nn.Linear(400, 300)
        self.bn1 = nn.LayerNorm(400)
        self.bn2 = nn.LayerNorm(300)
   

        self.mu = nn.Linear(300, action_space)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)  
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)   
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
            x = self.fc1(state)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = torch.tanh(self.mu(x))
            return x