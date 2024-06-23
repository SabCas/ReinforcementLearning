import gymnasium as gym
import collections
import numpy as np
import cv2
import ale_py.roms  # Ensure Atari ROMs are registe


# FireResetEnv Wrapper
class FireResetEnv(gym.Wrapper):
    def __init__(self, env, fire_reset=True):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        self.fire_reset = fire_reset

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fire_reset:
            obs, _, terminated, truncated, _ = self.env.step(1)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
            obs, _, terminated, truncated, _ = self.env.step(2)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)
        return obs, info

# MaxAndSkipEnv Wrapper
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if self._skip > 1:
                self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

# PreprocessFrame Wrapper
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84, 1)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def observation(self, frame):
        return self.process(frame)

    def process(self, frame):
        if self.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
        return frame.astype(np.uint8)



class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, buffer_size=4):
        super().__init__(env)
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(buffer_size, *env.observation_space.shape[1:]), dtype=np.uint8)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.buffer)
        obs, info = self.env.reset(**kwargs)    
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        self.buffer = np.squeeze(self.buffer)
        return np.copy(self.buffer)


# ImageToPytorch Wrapper
class ImageToPytorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPytorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(1, obs_shape[0], obs_shape[1]), dtype=np.float32
            )

    def observation(self, observation):
        if observation.ndim == 2:
            observation = np.expand_dims(observation, axis=-1)
        return np.moveaxis(observation, -1, 0)

# ScaledFloatFrame Wrapper
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

def make_env(env_name, render_mode='rgb_array'):
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env, fire_reset=False)
    env = PreprocessFrame(env, shape=(84, 84, 1))
    env = ImageToPytorch(env)
    env = BufferWrapper(env, buffer_size=4)
    env = ScaledFloatFrame(env)

          # Ensure environment renders to rgb_array for video recording
    
    return env




