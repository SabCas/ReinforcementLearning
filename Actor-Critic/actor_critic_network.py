import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

class Actor_Critic(nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(Actor_Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, 2048)
        self.fc2 = nn.Linear(2048, 1536)
        self.pi = nn.Linear(1536, action_space)
        self.v = nn.Linear(1536, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, state):
        # Shared network layers
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Actor output
        action_probs = F.softmax(self.pi(x), dim=-1)
        # Critic output
        state_value = self.v(x)
        
        return action_probs, state_value
              

