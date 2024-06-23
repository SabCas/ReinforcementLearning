
"""
This script is the main entry point for training a reinforcement learning agent using the Deep Q-Learning algorithm.

The script sets up the environment using the `make_env` function from the `wrapper` module, and creates an `Agent` instance from the `agent` module. It then trains the agent for a specified number of epochs using the `train` method of the `Agent` class.

The script assumes that a CUDA-enabled GPU is available, and will use it if possible. Otherwise, it will fall back to using the CPU.
"""
import torch
from wrapper import make_env
from agent import Agent
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()




def main():
    
    env = make_env('PongNoFrameskip-v4')
    agent = Agent(env, dqn_type='DuelingDQN', double_dqn=False)


    # Train the agent
    num_epochs = 500  
    agent.train(num_epochs)

if __name__ == '__main__':
    main()
