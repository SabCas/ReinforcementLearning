#!/usr/bin/env python3

import argparse
import numpy as np
import gymnasium as gym
import torch
import collections
import time
from dqn import DQN
from ale_py.env import AtariEnv
import wrapper
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics

import gym
from gym.wrappers import AtariPreprocessing

# Define environment name
env_name = 'PongNoFrameskip-v4'

# Create the environment with AtariPreprocessing wrapper


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

# Patch to allow rendering Atari games in ALE-Py
_original_atari_render = AtariEnv.render

def atari_render(self, mode='rgb_array'):
    original_render_mode = self.render_mode
    try:
        self.render_mode = mode
        return _original_atari_render(self)
    finally:
        self.render_mode = original_render_mode

AtariEnv.render = atari_render

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest='vis',
                        help="Disable visualization",
                        action='store_false')
    args = parser.parse_args()

    env = wrapper.make_env(args.env, render_mode='rgb_array')
    if args.record:
        env = RecordVideo(env, video_folder=args.record, name_prefix="eval")

    # Initialize DQN model
    net = DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)  # Load model state
    net.load_state_dict(state)  # Load model weights

    state, _ = env.reset()  # Reset environment and get initial state
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()

        env.render()  # Render the environment in 'human' mode
        state_v = torch.tensor(np.array([state], copy=False))  # Convert state to tensor
        q_vals = net(state_v).data.numpy()[0]  # Forward pass through the DQN
        action = np.argmax(q_vals)  # Choose action based on Q-values
        c[action] += 1

        state, reward, terminated, truncated, _ = env.step(action)  # Take action in the environment
        done  = terminated or truncated  # Check if episode is done     
        total_reward += reward  # Accumulate total reward

        if done:
            break

        delta = 1 / FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)  # Control frame rate

    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)

    if args.record:
        env.close()  # Close environment after recording video
