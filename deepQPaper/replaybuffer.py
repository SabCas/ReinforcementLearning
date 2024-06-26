"""
Implements a replay buffer for storing and sampling experiences in a reinforcement learning environment.

The `ReplayBuffer` class provides methods for adding experiences to the buffer and sampling a batch of experiences from the buffer.

Attributes:
    buffer (collections.deque): A deque that stores the experiences, with a maximum length specified by the `capacity` parameter.

Methods:
    add_sample(state, action, reward, done, new_state): Adds a new experience to the buffer.
    sample(batch_size): Samples a batch of experiences from the buffer.
    __len__(): Returns the number of experiences in the buffer.
"""
import numpy as np
import collections

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add_sample(self, state, action, reward, done, new_state):
        self.buffer.append(Experience(state, action, reward, done, new_state))
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)
    
    def __len__(self):
        return len(self.buffer)

