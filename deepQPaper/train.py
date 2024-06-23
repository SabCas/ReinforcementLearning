import torch
from wrapper import make_env
from pytorch_lightning.loggers import TensorBoardLogger
from agent import Agent
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()




def main():
    
    env = make_env('PongNoFrameskip-v4')
    agent = Agent(env)

    # Train the agent
    num_epochs = 500  
    agent.train(num_epochs)

if __name__ == '__main__':
    main()
