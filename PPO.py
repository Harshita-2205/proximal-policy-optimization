import tensorflow as tf
import numpy as np
import random

# Game environment setup
class GridWorldEnv:
    def __init__(self, size=5):
        # Initialize the grid world environment with a default size of 5x5
        self.size = size
        self.state = None
        self.target = (size - 1, size - 1)  # Set the target at the bottom-right corner
        self.reset()  # Initialize the environment state

    def reset(self):
        # Reset the environment to its initial state
        self.state = (0, 0)  # Starting position at the top-left corner
        # Randomly place 5 obstacles in the grid, avoiding the edges
        self.obstacles = {(random.randint(1, self.size - 2), random.randint(1, self.size - 2)) for _ in range(5)}
        return self.get_state()  # Return the flattened state representation

    def get_state(self):
        # Create a grid representation of the environment
        state = np.zeros((self.size, self.size))  # Initialize the grid with zeros
        state[self.state] = 1  # Mark the agent's current position
        state[self.target] = 0.5  # Mark the target position
        for obs in self.obstacles:
            state[obs] = -1  # Mark obstacles with -1
        return state.flatten()  # Return the flattened grid state

    def step(self, action):
        # Execute the agent's action and update the environment state
        x, y = self.state
        if action == 0: x = max(x - 1, 0)   # Move up
        elif action == 1: x = min(x + 1, self.size - 1)  # Move down
        elif action == 2: y = max(y - 1, 0)  # Move left
        elif action == 3: y = min(y + 1, self.size - 1)  # Move right

        self.state = (x, y)  # Update the agent's position

        # Check if the agent has reached the target
        if self.state == self.target:
            return self.get_state(), 1, True  # Return reward of 1 and mark episode as done
        # Check if the agent hit an obstacle
        elif self.state in self.obstacles:
            return self.get_state(), -1, True  # Return penalty of -1 and mark episode as done
        else:
            return self.get_state(), -0.1, False  # Small penalty for each step, episode continues

# PPO model
class ActorCritic(tf.keras.Model):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')  # Fully connected layer
        self.policy_logits = tf.keras.layers.Dense(action_size)  # Output layer for policy (actions)
        self.value = tf.keras.layers.Dense(1)  # Output layer for state value estimation

    def call(self, state):
        # Forward pass of the model
        x = self.dense1(state)  # Pass state through the dense layer
        logits = self.policy_logits(x)  # Compute logits for actions
        value = self.value(x)  # Compute state value
        return logits, value

# PPO Agent
class PPOAgent:
    def __init__(self, env, gamma=0.99, clip_ratio=0.2, lr=0.001):
        self.env = env  # Reference to the environment
        self.gamma = gamma  # Discount factor for rewards
        self.clip_ratio = clip_ratio  # PPO clipping ratio
        self.model = ActorCritic(action_size=4)  # Initialize Actor-Critic model with 4 actions
        self.optimizer = tf.keras.optimizers.Adam(lr)  # Adam optimizer
        # Initialize buffers to store trajectories
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []

    def get_action(self, state):
        # Choose an action based on the current state
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)  # Expand dimensions for batch processing
        logits, value = self.model(state)  # Get logits and state value
        action = tf.random.categorical(logits, 1)[0, 0]  # Sample action from the policy
        prob = tf.nn.softmax(logits)  # Compute action probabilities
        log_prob = tf.math.log(prob[0, action])  # Compute log probability of the chosen action
        return int(action.numpy()), float(log_prob), float(value)

    def store(self, state, action, reward, log_prob, value):
        # Store trajectory data for training
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def train(self):
        # Train the PPO model using stored trajectories
        returns, advs = self.compute_advantages()  # Compute returns and advantages
        self.update_model(returns, advs)  # Update model parameters

    def compute_advantages(self):
        # Compute returns and advantages for the trajectory
        returns, advs = [], []
        discounted_sum = 0
        for reward in self.rewards[::-1]:
            discounted_sum = reward + self.gamma * discounted_sum  # Compute discounted returns
            returns.insert(0, discounted_sum)  # Prepend to maintain chronological order
        returns = np.array(returns)
        advs = returns - np.array(self.values)  # Compute advantages
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)  # Normalize advantages
        return returns, advs

    def update_model(self, returns, advs):
        # Update the PPO model
        with tf.GradientTape() as tape:
            for i in range(len(self.states)):
                logits, value = self.model(tf.expand_dims(self.states[i], 0))  # Forward pass
                new_log_prob = tf.math.log(tf.nn.softmax(logits)[0, self.actions[i]])  # New log probability

                # Calculate importance sampling ratios
                ratio = tf.exp(new_log_prob - self.log_probs[i])
                surr1 = ratio * advs[i]  # Surrogate loss 1
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advs[i]  # Surrogate loss 2
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO clipped loss

                value_loss = tf.reduce_mean(tf.square(returns[i] - value))  # Value function loss
                entropy_bonus = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax(logits) * tf.math.log(tf.nn.softmax(logits) + 1e-10), axis=1))  # Entropy for exploration
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus  # Total loss

            grads = tape.gradient(loss, self.model.trainable_variables)  # Compute gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  # Apply gradients

        # Clear buffers after updating
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []

# Main training loop
env = GridWorldEnv(size=5)  # Initialize environment
agent = PPOAgent(env)  # Initialize PPO agent

for episode in range(500):  # Run for 500 episodes
    state = env.reset()  # Reset environment
    episode_reward = 0  # Track total reward for the episode

    while True:
        action, log_prob, value = agent.get_action(state)  # Get action from agent
        next_state, reward, done = env.step(action)  # Take action in the environment
        agent.store(state, action, reward, log_prob, value)  # Store trajectory data

        episode_reward += reward  # Accumulate reward
        state = next_state  # Update current state

        if done:  # Check if the episode is over
            agent.train()  # Train the model
            break  # Exit loop

    print(f"Episode: {episode+1}, Total Reward: {episode_reward}")  # Log episode results
