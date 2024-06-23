"""
The Agent class is responsible for managing the Deep Q-Learning (DQN) agent, including the Q-network, target network, replay buffer, and training logic.

The agent is initialized with the environment, and can be configured to use either a regular DQN or a Double DQN architecture. The agent's actions are selected using an epsilon-greedy strategy, where the agent either explores by taking a random action or exploits by taking the action that maximizes the predicted Q-value.

The agent's experience (state, action, reward, done, next state) is stored in a replay buffer, and the agent is trained by sampling batches from the replay buffer and minimizing the mean squared error between the expected and predicted Q-values.

The target network is periodically updated to stabilize the training process. The agent's training progress is logged using TensorBoard, and the agent's model is saved periodically.
"""
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from replaybuffer import ReplayBuffer
from dqn import DQN
import numpy as np
import time

GAMMA = 0.99
BATCH_SIZE = 32  
REPLAY_SIZE = 20000  
LEARNING_RATE = 1e-4  
SYNC_TARGET_FRAMES = 500  # Adjust target network sync frequency as needed
REPLAY_START_SIZE = 20000
SKIP_FRAMES = 4

EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY_LAST_FRAME = 150000
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')  





class Agent:
    def __init__(self, env, double_dqn=False):
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
        self.skip_frames = SKIP_FRAMES
        self.double_dqn = double_dqn
        
        self.writer = SummaryWriter()  
        
        while len(self.replay_buffer) < REPLAY_START_SIZE:
            self.play_step(epsilon=self.epsilon)

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.total_reward = 0.0
        return state, info
    
    def get_epsilon(self):
        epsilon = max(EPSILON_FINAL, EPSILON_START - self.global_step / EPSILON_DECAY_LAST_FRAME)
        return epsilon

    @torch.no_grad()
    def take_action(self, state, epsilon=0.0):
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()  # Explore: take a random action
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)
            q_vals_v = self.q_net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())  # Exploit: take the best known action
        
        return action

    def play_step(self, epsilon=0.0):
        skip_frames = self.skip_frames
        if self.state is None:
            self.state, _ = self.reset()

        done_reward = None

        # Take action based on epsilon-greedy strategy
        action = self.take_action(self.state, epsilon)
        accumulated_reward = 0.0
        is_done = False

        # Process multiple frames according to skip_frames
        for _ in range(skip_frames):
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            
            reward = np.clip(reward, -1.0, 1.0)  

            accumulated_reward += reward
            is_done = terminated or truncated

            if not is_done:
                self.state = new_state
            else:
                break

        self.total_reward += accumulated_reward

        exp = (self.state, action, accumulated_reward, is_done, new_state)
        self.replay_buffer.add_sample(*exp)
        self.state = new_state if not is_done else None

        if is_done:
            done_reward = self.total_reward
            self.episode_returns.append(self.total_reward)
            print('Episode return:', self.total_reward)
            self.total_reward = 0.0
            self.state, _ = self.reset()

        self.epsilon = self.get_epsilon()

        return done_reward



    
    def calc_loss(self, batch):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

        if self.double_dqn:
            # Double DQN: use online network to select actions and target network to evaluate Q-values
            q_vals_next = self.q_net(next_states_v)
            _, act_next = torch.max(q_vals_next, dim=1)
            act_next = act_next.unsqueeze(-1)
            q_vals_target = self.target_net(next_states_v)
            next_state_values = q_vals_target.gather(1, act_next).squeeze(-1)
        else:
            # Regular DQN: use target network for both action selection and Q-value evaluation
            q_vals_target = self.target_net(next_states_v)
            next_state_values = q_vals_target.max(1)[0]

        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * GAMMA + rewards_v

        q_vals = self.q_net(states_v)
        state_action_values = q_vals.gather(1, actions_v).squeeze(-1)

        # Calculate TD error
        td_error = state_action_values - expected_state_action_values

        # Clip the TD error
        clipped_td_error = torch.clamp(td_error, -1.0, 1.0)

        # Compute the loss with clipped TD error
        loss = torch.mean(clipped_td_error ** 2)

        return loss


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

        self.writer.close()  


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