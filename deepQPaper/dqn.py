import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        print('input shapeeeeeeeeeeeeeeeeeeeeeee', input_shape)
        # Define convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Calculate the output size of the convolutions
        conv_out_size = self.get_conv_out(input_shape)


        # Define fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def get_conv_out(self, image_dim):
        output = self.conv(torch.rand(1, *image_dim))
        output_size = np.prod(output.size())
        return output_size


    def forward(self, inp):
        # inp is expected to have shape [batch_size, channels, height, width]
        x = self.conv(inp)
        x = x.view(inp.size(0), -1)  # Flatten the output from convolutions
        x = self.fc(x)
        return x
