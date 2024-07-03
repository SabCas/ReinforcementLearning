import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, TimeLimit
from torch.utils.tensorboard import SummaryWriter as tf
from agent import Agent
import numpy as np


"""
Computes the average rewards every 100 episodes and logs the initial values and average reward to TensorBoard.

Args:
    rewards (list): A list of rewards for each episode.
    interval (int): The interval at which to compute the average rewards (default is 100).

Returns:
    list: A list of average rewards for each interval.
"""
def compute_average_rewards(rewards, interval=100):
    avg_rewards = []
    for i in range(0, len(rewards), interval):
        avg_reward = np.mean(rewards[i:i+interval])
        avg_rewards.append(avg_reward)
    return avg_rewards

"""
Logs the initial values (learning rate and gamma) and the average reward to TensorBoard.

Args:
    log_dir (str): The directory to save the TensorBoard logs.
    lr_init (float): The initial learning rate.
    gamma_init (float): The initial gamma value.
    avg_rewards (list): A list of average rewards for each interval.
"""
def log_initial_values_and_avg_reward(log_dir, lr_init, gamma_init, avg_rewards):
    writer = tf.summary.create_file_writer(log_dir)

    # Log initial values
    with writer.as_default():
        tf.summary.scalar('Initial Learning Rate', lr_init, step=0)
        tf.summary.scalar('Initial Gamma', gamma_init, step=0)

        # Log average reward
        for i, avg_reward in enumerate(avg_rewards):
            tf.summary.scalar('Average Reward', avg_reward, step=i)

    writer.close()




 
"""
    Runs the main training loop for the REINFORCE agent on the LunarLander-v3 environment.
    
    The function creates the environment, wraps it with a TimeLimit and RecordVideo wrapper, creates an Agent instance, and then runs the training loop for 3000 episodes.
    
    During the training loop, the agent chooses an action, steps the environment, stores the reward, and then trains the agent. The average score over the last 100 episodes is printed after each episode.
    
    After the training loop, the initial values (learning rate and gamma) and the average reward over the training run are logged to TensorBoard.
    
    Args:
        None
    
    Returns:
        None
    """
def main():

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    env = TimeLimit(env, max_episode_steps=400)
    env = RecordVideo(env, video_folder='videos/', episode_trigger=lambda episode_id: episode_id % 50 == 0)
    agent = Agent(gamma=0.99, lr=0.0005, action_space=env.action_space.n, state_space=env.observation_space.shape[0])

    n_games = 3000
    avg_rewards = []
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
            agent.store_reward(reward)
            observation = new_observation
        agent.train()
        scores.append(score)
        print(f"Game {i}: Average score: {np.mean(scores[-100:])} Reward: {score}")
    log_initial_values_and_avg_reward(log_dir='logs', lr_init=agent.lr, gamma_init=agent.gamma, avg_rewards=compute_average_rewards(scores))

    env.close()
            
    

if __name__ == "__main__":
    main()
