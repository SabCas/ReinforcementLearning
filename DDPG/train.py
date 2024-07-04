from agent import Agent
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
from torch.utils.tensorboard import SummaryWriter as tf
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

if __name__=="__main__":
    env = gym.make("LunarLanderContinuous-v3", render_mode='rgb_array')
    env = TimeLimit(env, max_episode_steps=400)
    env = RecordVideo(env, video_folder='videos/', episode_trigger=lambda episode_id: episode_id % 50 == 0)
    agent = Agent(lr_actor=0.001, lr_critic=0.001, gamma=0.99, tau=0.001,
                   state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    best_score = -np.inf
    score_list = []
    n_games = 1000
    for epoch in range(n_games):
        state,_ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_history(state, action, reward, new_state, done)
            agent.train()
            state = new_state
            total_reward += reward
        score_list.append(total_reward)
        avg_reward = np.mean(score_list[-100:])
        print(f"Episode: {epoch+1}, Score: {total_reward}, Average Score: {avg_reward}")
    log_initial_values_and_avg_reward(log_dir='logs', lr_init=agent.lr, gamma_init=agent.gamma, avg_rewards=compute_average_rewards(score_list))

    env.close()