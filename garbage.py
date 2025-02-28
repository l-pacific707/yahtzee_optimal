import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# -------------------------
# 1. ENVIRONMENT (SKELETON)
# -------------------------
class YahtzeeEnv(gym.Env):
    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        # Define spaces (replace with actual spaces for states and actions)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)  # e.g., 5 possible actions

    def reset(self):
        # Reset environment state (dice, categories, etc.)
        self.state = np.zeros(10, dtype=np.float32)
        # Return initial observation
        return self.state

    def step(self, action):
        # 1) Apply action (roll dice or choose category, etc.)
        # 2) Calculate reward
        # 3) Determine if episode is done
        # 4) Return the next state, reward, done, and optionally info dict
        next_state = self.state  # Placeholder for next state calculation
        reward = 0.0            # Placeholder for reward calculation
        done = False            # Placeholder for done condition
        info = {}

        # Example placeholder: random update just for demonstration
        next_state = next_state + np.random.rand(10) * 0.1

        return next_state, reward, done, info

# -------------------------
# 2. Q-NETWORK DEFINITION
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# 3. REPLAY BUFFER
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------------
# 4. TRAINING LOOP
# -------------------------
def train_dqn(env, num_episodes=1000, batch_size=32, gamma=0.99,
              lr=1e-3, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):

    # Initialize Q-network & target network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(state.unsqueeze(0))
                    action = q_values.argmax().item()

            # Step environment
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)
            total_reward += reward

            # Store transition in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # Train the network if the buffer has enough samples
            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Convert to tensors
                states_t = torch.stack(states)
                actions_t = torch.LongTensor(actions)
                rewards_t = torch.FloatTensor(rewards)
                next_states_t = torch.stack(next_states)
                dones_t = torch.BoolTensor(dones)

                # Compute current Q-values
                q_values = q_net(states_t)
                q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze()

                # Compute target Q-values
                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0]
                    target_q = rewards_t + gamma * max_next_q * (~dones_t)

                # MSE loss
                loss = nn.MSELoss()(q_values, target_q)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network periodically (e.g., every 50 steps)
            # You could also do it every certain number of episodes
            if random.random() < 0.01:
                target_net.load_state_dict(q_net.state_dict())

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        print(f"Episode {episode} | Total Reward: {total_reward}")

    return q_net

# -------------------------
# 5. MAIN SCRIPT
# -------------------------
if __name__ == "__main__":
    env = YahtzeeEnv()
    trained_q_net = train_dqn(env, num_episodes=50)
    # Use 'trained_q_net' for inference or further analysis
