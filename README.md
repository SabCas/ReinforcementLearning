# Reinforcement Learning
This repository contains a collection of reinforcement learning algorithms implemented in Python. The algorithms are designed to solve various reinforcement learning problems, including classic control tasks and Atari games.

## Algorithms
The following algorithms are implemented in this repository:

## Deep Q-Learning (DQN)

Implementation of the Deep Q-Learning algorithm for learning control policies from high-dimensional sensory inputs.
Includes support for experience replay, target network updates, and exploration strategies.
 - Dueling Deep Q-Network (Dueling DQN)

    An extension of the DQN algorithm that separates the estimation of state values and advantages, leading to better performance.
- Double Deep Q-Network (Double DQN)

    A variant of the DQN algorithm that addresses the overestimation problem by using a separate target network to evaluate the Q-values.
## REINFORCE (Monte Carlo Policy Gradient)

An implementation of the REINFORCE algorithm, which uses Monte Carlo methods to estimate the policy gradient and update the policy parameters.
## Actor-Critic
An implementation of the Actor-Critic algorithm, which combines an actor network for policy estimation and a critic network for value estimation.
Usage
