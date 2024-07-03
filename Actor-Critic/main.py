"""
This module contains the main entry point for the Actor-Critic reinforcement learning agent.

The `main()` function sets up the environment, creates an Agent instance, and runs the training loop for a specified number of games. It also logs the initial values of the learning rate and discount factor, as well as the average reward per episode, to TensorBoard.

The `compute_average_rewards()` function calculates the average reward over a specified interval of episodes.

The `log_initial_values_and_avg_reward()` function creates a TensorBoard log directory and logs the initial values and average reward per episode.
"""
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
from torch.utils.tensorboard import SummaryWriter as tf
from agent import Agent
import numpy as np
import os 





    # Function to compute average rewards every 100 episodes
def compute_average_rewards(rewards, interval=100):
    avg_rewards = []
    for i in range(0, len(rewards), interval):
        avg_reward = np.mean(rewards[i:i+interval])
        avg_rewards.append(avg_reward)
    return avg_rewards

# Function to log initial values and average reward
def log_initial_values_and_avg_reward(log_dir, lr_init, gamma_init, avg_rewards):
    os.makedirs(log_dir, exist_ok=True)
    writer = tf(log_dir=log_dir)

    # Log initial values
    with writer.as_default():
        tf.summary.scalar('Initial Learning Rate', lr_init, step=0)
        tf.summary.scalar('Initial Gamma', gamma_init, step=0)

        # Log average reward
        for i, avg_reward in enumerate(avg_rewards):
            tf.summary.scalar('Average Reward', avg_reward, step=i)

    writer.close()

def main():
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env = TimeLimit(env, max_episode_steps=400)
    env = RecordVideo(env, video_folder='videos/', episode_trigger=lambda episode_id: episode_id % 50 == 0)
    agent = Agent(gamma=0.99, lr=5e-6, action_space=env.action_space.n, state_space=env.observation_space.shape[0])

    n_games = 3000
    scores = []
    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.train(observation, reward, new_observation, done)
            observation = new_observation
        scores.append(score)
        print(f"Game {i}: Average score: {np.mean(scores[-100:])} Reward: {score}")
    log_initial_values_and_avg_reward(log_dir='logs', lr_init=agent.lr, gamma_init=agent.gamma, avg_rewards=compute_average_rewards(scores))

    env.close()
            
    

if __name__ == "__main__":
    main()