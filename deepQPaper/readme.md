# Deep Q-Learning for Atari Games

This project implements Deep Q-Learning, a reinforcement learning algorithm, to train an agent to play Atari games. The agent uses a Deep Q-Network (DQN) or a Dueling DQN to approximate the Q-values for each state-action pair.

## Features

- Implementation of the DQN and Dueling DQN architectures
- Experience replay buffer for efficient training
- Double DQN variant for improved stability
- Tensorboard logging for monitoring training progress
- Evaluation and visualization of trained agents

## Requirements

- Python 3.6 or higher
- PyTorch
- Gymnasium (formerly OpenAI Gym)
- OpenCV
- NumPy
- Matplotlib (for visualization)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/deep-q-learning-atari.git
cd deep-q-learning-atari
```

## Project Structure
agent.py: Contains the Agent class for training and interacting with the environment.
dqn.py: Defines the DQN model architecture.
duelingDQN.py: Defines the Dueling DQN model architecture.
replaybuffer.py: Implements the experience replay buffer.
wrapper.py: Contains environment wrappers for preprocessing observations.
train.py: Entry point for training the agent.
play.py: Entry point for evaluating a trained agent.
Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
