# Actor-Critic Reinforcement Learning
This project implements an Actor-Critic algorithm for solving reinforcement learning problems in in the Lunar Lander environment. The Actor-Critic method combines the advantages of both policy-based and value-based methods, allowing for efficient learning and exploration.

## Overview
The Actor-Critic algorithm consists of two main components:

- Actor: The actor is responsible for learning the policy function, which maps states to actions. It determines the action to take in a given state.

- Critic: The critic is responsible for learning the value function, which estimates the expected future reward for a given state or state-action pair. It provides feedback to the actor, helping it learn a better policy.

The actor and critic work together in an iterative process. The actor takes actions based on the current policy, and the critic evaluates the resulting states and rewards. The critic then updates the value function and provides feedback to the actor, allowing the actor to update its policy accordingly.

## Installation
Clone the repository:
```
git clone https://github.com/SabCas/ReinforcementLearning.git
``

Navigate to the Actor-Critic directory:
```
cd ReinforcementLearning/Actor-Critic
```

Install requirements

```
pip install -r requirements.txt
```

Run the main script
```
python main.py
```
