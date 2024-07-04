import numpy as np

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# # In your DDPG agent class or wherever actions are selected
# noise = OrnsteinUhlenbeckNoise(action_dim)

# def select_action(self, state, policy_network):
#     action = policy_network(state)  # Get action from policy network
#     action += noise.noise()  # Add noise to action for exploration
#     return action
