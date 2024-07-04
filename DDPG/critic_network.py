
"""
The CriticNetwork class is a PyTorch module that represents the critic network in a Deep Deterministic Policy Gradient (DDPG) reinforcement learning algorithm.

The critic network is responsible for estimating the value function, which represents the expected future reward for a given state and action. The critic network takes the current state and the selected action as input, and outputs a scalar value representing the estimated value of that state-action pair.

The network architecture consists of two fully connected layers with layer normalization, followed by a final fully connected layer that outputs the estimated value. The weights of the network are initialized using a uniform distribution.

The class also includes an Adam optimizer for training the network parameters.
"""
import torch
import numpy as np
import torch.nn.functional as F 

class CriticNetwork(torch.nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(CriticNetwork, self).__init__()
        self.lr = lr
        self.fc1 = torch.nn.Linear(state_space , 400)
        self.fc2 = torch.nn.Linear(400, 300)
        self.bn1 = torch.nn.LayerNorm(400)
        self.bn2 = torch.nn.LayerNorm(300)

        self.action_value =  torch.nn.Linear(action_space, 300)

        self.q = torch.nn.Linear(300, 1)
        #calculates the squared rooot of hte input feature \xavier initialiization or glorot
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])  
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])  
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)  


        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        


    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        action_value = self.action_value(action)
        state_action_value = F.relu(torch.add(x, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value