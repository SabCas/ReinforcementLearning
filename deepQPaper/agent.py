"""
The `Agent` class is the main reinforcement learning agent that interacts with the environment, stores experiences in a replay buffer, and trains a deep Q-network (DQN) model.

The agent has the following key responsibilities:
- Initialize the Q-network and target network, as well as the optimizer and replay buffer.
- Implement the `play_step` method to interact with the environment, store experiences, and update the agent's state.
- Implement the `calc_loss` method to compute the loss for a batch of experiences, used for training the Q-network.
- Implement the `update_target_network` method to periodically update the target network with the latest weights from the Q-network.
- Implement the `train` method to train the agent for a specified number of epochs, including logging metrics to TensorBoard.
"""
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from shimmy import GymV21CompatibilityV0
from torch.utils.tensorboard import SummaryWriter
from replaybuffer import ReplayBuffer
from dqn import DQN
import time

GAMMA = 0.99
BATCH_SIZE = 64  
REPLAY_SIZE = 20000  
LEARNING_RATE = 1e-4  
SYNC_TARGET_FRAMES = 500  # Adjust target network sync frequency
REPLAY_START_SIZE = 20000
SKIP = 4

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150000

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')  





class Agent:
    def __init__(self, env):
        self.env = env
        self.q_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        self.target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        
        self.replay_buffer = ReplayBuffer(REPLAY_SIZE)
        self.total_reward = 0.0
        self.state = None
        self.epsilon = EPSILON_START
        self.episode_returns = []
        self.global_step = 0
        
        self.writer = SummaryWriter()  
        self.skip = SKIP  # Number of frames to skip
        
        while len(self.replay_buffer) < REPLAY_START_SIZE:
            self.play_step(epsilon=self.epsilon)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.total_reward = 0.0
        return state, info
    
    def get_epsilon(self):
        epsilon = max(EPSILON_FINAL, EPSILON_START - self.global_step / EPSILON_DECAY_LAST_FRAME)
        return epsilon

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)
            q_vals_v = self.q_net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            return int(act_v.item())

    def play_step(self):
        if self.state is None:
            self.state, _ = self.reset()

        action = self.select_action(self.state)
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        clipped_reward = np.clip(reward, -1.0, 1.0)
        is_done = terminated or truncated
        self.total_reward += clipped_reward
        exp = Experience(self.state, action, clipped_reward, is_done, new_state)
        self.replay_buffer.add_sample(*exp)
        self.state = new_state

        if is_done:
            self.episode_returns.append(self.total_reward)
            print('Episode return:', self.total_reward)
            self.state = None
            self.total_reward = 0.0
            self.state, _ = self.reset()

        self.epsilon = self.get_epsilon()
        self.global_step += 1
    
    def calc_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
        
        state_action_values = self.q_net(states_v).gather(1, actions_v).squeeze(-1)
        next_state_values = self.target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * GAMMA + rewards_v

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            episode_return = self.train_epoch(epoch)
            self.update_target_network()
            
            mean_reward = np.mean(self.episode_returns[-100:])
            print(f"Epoch {epoch}: episode_return={episode_return:.3f}, mean_reward={mean_reward:.3f}, epsilon={self.epsilon:.3f}")
            
            self.writer.add_scalar('episode/return', episode_return, self.global_step)
            self.writer.add_scalar('episode/mean_reward', mean_reward, self.global_step)
            self.writer.add_scalar('epsilon', self.epsilon, self.global_step)
            
            if (epoch + 1) % 100 == 0:  # Save every 100 epochs
                torch.save(self.q_net.state_dict(), f'model_epoch_{epoch}.pth')

        self.writer.close()  # Close TensorBoard writer at the end of training


    def train_epoch(self, epoch):
        episode_return = None
        episode_steps = 0

        while len(self.replay_buffer) < REPLAY_START_SIZE:
            print(f"Replay buffer is too small ({len(self.replay_buffer)} / {REPLAY_START_SIZE}), waiting for more samples...")
            time.sleep(1)

        while episode_return is None:
            episode_return = self.play_step(epsilon=self.epsilon)
            self.global_step += 1
            episode_steps += 1
            self.epsilon = self.get_epsilon()
            
            if len(self.replay_buffer) >= BATCH_SIZE:
                batch = self.replay_buffer.sample(BATCH_SIZE)
                loss = self.calc_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.writer.add_scalar('train_loss', loss.item(), self.global_step)

                if self.global_step % SYNC_TARGET_FRAMES == 0:
                    self.update_target_network()

        self.episode_returns.append(episode_return)
        print(f"Epoch {epoch}: Episode completed with return {episode_return}, total steps {episode_steps}")
        
        return episode_return

