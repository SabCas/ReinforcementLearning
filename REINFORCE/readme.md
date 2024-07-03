# Reinforcement Learning: REINFORCE Algorithm
This repository contains an implementation of the REINFORCE algorithm, a policy gradient method for reinforcement learning. The REINFORCE algorithm is used to train an agent to solve the LunarLander-v3 environment from the Gymnasium library (formerly OpenAI Gym).

## Overview
The REINFORCE algorithm is a Monte Carlo policy gradient method that directly optimizes the policy function by computing gradients of the expected reward with respect to the policy parameters. It belongs to the family of policy gradient methods, which aim to find the optimal policy that maximizes the expected cumulative reward.

In this implementation, we use a neural network to represent the policy function, which takes the state as input and outputs the probability distribution over actions. The network is trained using the REINFORCE algorithm, which updates the network parameters based on the rewards received during episodes.

## Repository Structure
- main.py: The main entry point of the project. It sets up the environment, creates the agent, and runs the training loop.
- agent.py: Contains the Agent class, which implements the REINFORCE algorithm and interacts with the environment.
- policy_network.py: Defines the neural network architecture used for the policy function.
- videos/: Directory to store recorded videos of the agent's performance (generated every 50 episodes).
- logs/: Directory to store TensorBoard logs for monitoring training progress.

## Usage
Clone the repository:
```
git clone https://github.com/SabCas/ReinforcementLearning.git
cd ReinforcementLearning/REINFORCE
```

## Monitoring Training Progress
You can monitor the training progress using TensorBoard. After running the main.py script, open a new terminal and run:
```
tensorboard --logdir=logs
```

## Install the required packages by running:

```
pip install -r requirements.txt
```
