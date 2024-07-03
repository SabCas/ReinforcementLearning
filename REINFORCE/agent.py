"""
The `Agent` class is responsible for managing the reinforcement learning agent's policy network, action selection, and training.

Attributes:
    state_space (int): The size of the state space.
    action_space (int): The size of the action space.
    lr (float): The learning rate for the policy network.
    gamma (float): The discount factor for future rewards.
    policy (PolicyNetwork): The policy network used by the agent.
    reward_memory (list): A list of rewards collected during an episode.
    action_memory (list): A list of log probabilities of actions taken during an episode.

Methods:
    choose_action(obs: np.ndarray) -> int:
        Selects an action based on the current observation using the policy network.
    store_reward(reward: float):
        Stores a reward in the reward memory.
    train():
        Trains the policy network using the collected rewards and actions.
"""
from policy_network import PolicyNetwork
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_space, action_space, lr, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.policy = PolicyNetwork(state_space, action_space, lr)
        self.reward_memory = []
        self.action_memory = []


    def choose_action(self, obs):
        obs = torch.from_numpy(obs).float().to(device)
        prob = self.policy(obs) 
        action_prob = torch.distributions.Categorical(prob)
        action = action_prob.sample()
        log_prob = action_prob.log_prob(action)  
        self.action_memory.append(log_prob)
        return action.item()
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)   

    def train(self):
        G_t = np.zeros_like(self.reward_memory)
        total_reward = 0
        for t in reversed(range(len(self.reward_memory))):
            total_reward = self.reward_memory[t]+ self.gamma * total_reward
            G_t[t] = total_reward
        G_t = torch.tensor(G_t, dtype=torch.float).to(device)

        loss = 0
        for log_prob, G_t_value in zip(self.action_memory, G_t):
            loss +=(-log_prob * G_t_value)
            

        self.policy.optimizer.zero_grad()
        loss.backward()


        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

            