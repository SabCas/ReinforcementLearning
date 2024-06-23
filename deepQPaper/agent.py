"""
The `Agent` class is the main reinforcement learning agent that interacts with the environment, stores experiences in a replay buffer, and trains a deep Q-network (DQN) or a dueling DQN model.

The agent supports two types of DQN models: the standard DQN and the dueling DQN. It also supports the double DQN variant, which uses a separate target network to evaluate the Q-values.

The agent's main responsibilities include:
- Initializing the Q-network and target network
- Interacting with the environment and storing experiences in the replay buffer
- Calculating the loss for a batch of experiences and updating the Q-network
- Periodically updating the target network to match the Q-network
- Logging training progress to TensorBoard

The agent's training process involves playing episodes, collecting experiences, and updating the Q-network. The training continues for the specified number of epochs, with the target network being updated periodically to stabilize the learning process.
"""
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from replaybuffer import ReplayBuffer
from dqn import DQN
from duelingDQN import DuelingDQN  
import time

GAMMA = 0.99
BATCH_SIZE = 64  
REPLAY_SIZE = 20000  
LEARNING_RATE = 1e-4  
SYNC_TARGET_FRAMES = 500  
REPLAY_START_SIZE = 20000

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150000
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')


class Agent:
    def __init__(self, env, dqn_type='DQN', double_dqn=False):
        self.env = env
        self.dqn_type = dqn_type
        self.double_dqn = double_dqn

        if self.dqn_type == 'DQN':
            self.q_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
            self.target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        elif self.dqn_type == 'DuelingDQN':
            self.q_net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
            self.target_net = DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        else:
            raise ValueError(f"Unsupported DQN type: {dqn_type}")

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        
        self.replay_buffer = ReplayBuffer(REPLAY_SIZE)
        self.total_reward = 0.0
        self.state = None
        self.epsilon = EPSILON_START
        self.episode_returns = []
        self.global_step = 0
        
        self.writer = SummaryWriter()  
        
        while len(self.replay_buffer) < REPLAY_START_SIZE:
            self.play_step()

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.total_reward = 0.0
        return state, info
    
    def get_epsilon(self):
        epsilon = max(EPSILON_FINAL, EPSILON_START - self.global_step / EPSILON_DECAY_LAST_FRAME)
        return epsilon

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)
            q_vals_v = self.q_net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        return action

    @torch.no_grad()
    def play_step(self):
        if self.state is None:
            self.state, _ = self.reset()

        done_reward = None
        action = self.take_action(self.state)
        
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward
        exp = (self.state, action, reward, is_done, new_state)
        self.replay_buffer.add_sample(*exp)
        self.state = new_state

        if is_done:
            done_reward = self.total_reward
            self.episode_returns.append(self.total_reward)
            print('Episode return:', self.total_reward)
            self.state = None
            self.total_reward = 0.0
            self.state, _ = self.reset()

        return done_reward
    
    def calc_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
        
        if self.double_dqn:
            # Double DQN: use q_net to select actions, target_net to evaluate Q values
            q_vals_next = self.q_net(next_states_v)
            _, act_v = torch.max(q_vals_next, dim=1)
            actions_next_v = act_v.unsqueeze(-1)
            next_state_values = self.target_net(next_states_v).gather(1, actions_next_v).squeeze(-1)
        else:
            # Regular DQN: both q_net and target_net for selecting actions and evaluating Q values
            next_state_values = self.target_net(next_states_v).max(1)[0]
        
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * GAMMA + rewards_v

        state_action_values = self.q_net(states_v).gather(1, actions_v).squeeze(-1)

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
            
            if (epoch + 1) % 100 == 0: 
                torch.save(self.q_net.state_dict(), f'model_epoch_{epoch}.pth')

        self.writer.close() 

    def train_epoch(self, epoch):
        episode_return = None
        episode_steps = 0

        while len(self.replay_buffer) < REPLAY_START_SIZE:
            print(f"Replay buffer is too small ({len(self.replay_buffer)} / {REPLAY_START_SIZE}), waiting for more samples...")
            time.sleep(1)

        while episode_return is None:
            episode_return = self.play_step()
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
