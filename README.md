# PPO GridWorld

This project demonstrates the implementation of Proximal Policy Optimization (PPO) to solve a simple GridWorld environment using TensorFlow.

## Description

The goal of the agent is to navigate a grid world and reach the target position while avoiding obstacles. The agent receives a reward for reaching the target and penalties for hitting obstacles or taking unnecessary steps.

PPO is a policy gradient reinforcement learning algorithm that uses a clipped surrogate objective function to improve the policy's performance. This implementation utilizes an actor-critic architecture, where the actor learns the policy and the critic estimates the value function.

## Requirements

- Python 3.9 or higher
- TensorFlow 2.12.0
- NumPy 1.24.3

You can install these dependencies using the provided `requirements.txt` file:
`bash pip install -r requirements.txt`

## Code Explanation

The code is structured into four main parts:

**1. Imports and Environment Setup:**

- Imports necessary libraries like TensorFlow, NumPy, and random.
- Defines the `GridWorldEnv` class to simulate the environment, handling state representation, actions, rewards, and episode termination.

**2. PPO Model (Actor-Critic):**

- Implements the `ActorCritic` class, which uses a neural network to represent both the policy (actor) and value function (critic).
- The `call` method defines the forward pass of the network.

**3. PPO Agent:**

- Implements the `PPOAgent` class, responsible for interacting with the environment, storing experiences, and training the model using the PPO algorithm.
- Key methods include `get_action`, `store`, `train`, `compute_advantages`, and `update_model`.

**4. Main Training Loop:**

- Initializes the environment and agent.
- Iterates through episodes, allowing the agent to explore and learn.
- Stores experiences and periodically updates the model's parameters.

**Overall Flow:**

1. The agent interacts with the environment, taking actions based on its policy.
2. It observes the rewards and stores the experiences in memory.
3. Periodically, the agent updates its policy and value function using the stored experiences and the PPO algorithm.
4. This process repeats until the agent learns to solve the task efficiently.

## Folder Structure
```
ppo-gridworld
├── README.md          # Project documentation
├── prompts.md         # Additional project notes or prompts
├── requirements.txt   # Dependencies for the project
└── Code
     ├── PPO.py        # Main PPO implementation

```
## Results

The notebook outputs the total reward obtained by the agent in each episode. You should observe the agent's performance improving over time as it learns to navigate the grid world more efficiently.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.
