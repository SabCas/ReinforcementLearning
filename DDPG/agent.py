import numpy as np
import torch as T
from ou_network import OrnsteinUhlenbeckNoise
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer
import torch.nn.functional as F

class Agent():
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, tau, gamma=0.99, capacity=100000, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size


        
        self.memory = ReplayBuffer(capacity)
        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_dim), action_dim=action_dim)

        self.actor = ActorNetwork(state_dim, action_dim, lr_actor)
        self.critic = CriticNetwork(state_dim, action_dim, lr_critic)
        self.target_actor = ActorNetwork(state_dim, action_dim, lr_actor)
        self.target_critic = CriticNetwork(state_dim, action_dim, lr_critic)

        self.update_network_parameters(tau)

    def select_action(self, state):
        self.actor.eval()
        state = T.tensor(state, dtype=T.float32).to(self.actor.device)
        mu = self.actor(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise.noise()).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()
    
    def store_history(self, state, action, reward, next_state, done):
        self.memory.add_sample(state, action, reward, next_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_params = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * self.actor.state_dict()[name].clone() + (1 - tau) * target_actor_state_dict[name].clone()

        for name in critic_state_dict:
            critic_state_dict[name] = tau * self.critic.state_dict()[name].clone() + (1 - tau) * target_critic_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)
        self.target_critic.load_state_dict(critic_state_dict)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = T.tensor(state_batch).float().to(self.actor.device)
        next_state_batch = T.tensor(next_state_batch).float().to(self.actor.device)
        action_batch = T.tensor(action_batch).float().to(self.actor.device)
        reward_batch = T.tensor(reward_batch).float().to(self.actor.device)
        done_batch = T.tensor(done_batch, dtype=T.bool).to(self.actor.device)

        target_actions = self.target_actor.forward(next_state_batch)
        critic_value_next = self.target_critic.forward(next_state_batch, target_actions)
        critic_value_next[done_batch] = 0.0
        critic_value_next = critic_value_next.view(-1)

        target = reward_batch + self.gamma * critic_value_next
        target = target.view(self.batch_size, 1)

        critic_value = self.critic.forward(state_batch, action_batch)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_value, target)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()
        
        self.update_network_parameters()










            
        

