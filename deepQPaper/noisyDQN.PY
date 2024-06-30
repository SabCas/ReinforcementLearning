"""
Implements a Noisy Linear layer for use in a Noisy DQN model.

The NoisyLinear layer adds noise to the weights and biases of a linear layer,
which can help with exploration in reinforcement learning tasks.

Args:
    input_shape (int): The size of the input to the linear layer.
    num_actions (int): The number of outputs (actions) of the linear layer.
    sigma (float): The standard deviation of the noise to be added to the weights and biases.
    device (torch.device): The device (CPU or GPU) to use for the layer.

Returns:
    torch.Tensor: The output of the noisy linear layer.
"""
from torch import nn
import torch
from torch.nn import functional as F

class NoisyLinear(nn.Module):
    def __init__(self, input_shape, num_actions,sigma, device):
        super(NoisyLinear, self).__init__()
        self.w_mu = nn.Parameter(torch.empty(num_actions, input_shape))
        self.w_sigma = nn.Parameter(torch.empty(num_actions, input_shape))
        self.b_mu = nn.Parameter(torch.empty(num_actions))
        self.b_sigma = nn.Parameter(torch.empty(num_actions))

    def forward(self,x, sigma=0.5):
        w_noise = torch.normal(0, sigma, size=self.w_mu.size).to(self.device)
        b_noise = torch.normal(0, sigma, size=self.b_mu.size).to(self.device)
        return F.linear(x, self.w_mu  + self.b_mu * w_noise, self.b_sigma + self.b_sigma * b_noise)


        