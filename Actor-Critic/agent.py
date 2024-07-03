from actor_critic_network import Actor_Critic
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_space, action_space, lr, gamma=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.actor_critic= Actor_Critic(state_space, action_space, lr)
        self.log_prob = None



    def choose_action(self, obs):
        obs = torch.from_numpy(obs).float().to(device)
        action_probs,_ = self.actor_critic.forward(obs)
        action_probs = torch.distributions.Categorical(action_probs)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)  
        self.log_prob = log_prob
        return action.item()
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)   

    def train(self, state, reward, new_state, done):
        state = torch.from_numpy(state).float().to(device)
        new_state = torch.from_numpy(new_state).float().to(device)  
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        _, critic_value = self.actor_critic.forward(state)
        _, new_critic_value = self.actor_critic.forward(new_state)

        delta = reward + self.gamma * new_critic_value * (1 - done) - critic_value
        actor_loss = -self.log_prob * delta
        critic_loss = 0.5 * delta.pow(2)

        loss = actor_loss + critic_loss
        self.actor_critic.optimizer.zero_grad()
        loss.backward()


        self.actor_critic.optimizer.step()


            