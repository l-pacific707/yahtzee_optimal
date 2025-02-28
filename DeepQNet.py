import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQNet(nn.Module):
    def __init__(self, state_dim=45, action_dim=43):
        super(DQNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.010, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100, rng=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        self.memory = deque(maxlen=buffer_size)
        #Cuda device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNet(state_dim, action_dim).to(self.device)
        self.target_net = DQNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr = lr, alpha = 0.99)
        self.rng = np.random.default_rng() if rng is None else rng
        
            
    def select_action(self, state, valid_actions):
        """
        Choose an action using an epsilon-greedy strategy,
        but only among the valid actions provided.
        """
        self.steps_done += 1
        # 1) With probability epsilon, pick a random valid action.
        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        else:
            # 2) Otherwise, pick the best valid action based on Q-values from the policy_net.
            with torch.no_grad():
                # Move input state to the same device as the network.
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Forward pass (no_grad context, so no gradient tracking).
                q_values = self.policy_net(state_tensor).squeeze(0)  # Shape: (action_dim,)

                # Create a mask where all actions are -inf except valid ones.
                masked_q = torch.full((self.action_dim,), float('-inf'), device=self.device)
                masked_q[valid_actions] = q_values[valid_actions]

                # Argmax on GPU; then convert to a Python int.
                action = int(torch.argmax(masked_q))
            
            return action


    def push_memory(self, transition):
        # Transition: (state, action, reward, next_state, done, valid_actions_next)
        self.memory.append(transition)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of transitions from memory
        indices = self.rng.choice(len(self.memory), self.batch_size, replace=False)
        transitions = [self.memory[i] for i in indices]

        # transitions is a list of tuples:
        # (state, action, reward, next_state, done, valid_actions_next)

        # Convert lists to NumPy arrays
        states = np.array([t[0] for t in transitions], dtype=np.float32)
        actions = np.array([t[1] for t in transitions], dtype=np.int64)
        rewards = np.array([t[2] for t in transitions], dtype=np.float32)
        next_states = np.array([t[3] for t in transitions], dtype=np.float32)
        dones = np.array([t[4] for t in transitions], dtype=np.float32)
        valid_actions_next = [t[5] for t in transitions]

        # Convert to PyTorch tensors
        batch_state = torch.from_numpy(states).to(self.device)
        batch_action = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        batch_reward = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        batch_next_state = torch.from_numpy(next_states).to(self.device)
        batch_done = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        # 1) Current Q-values for the taken actions (policy net)
        q_values_all = self.policy_net(batch_state)            # [batch_size, action_dim]
        current_q = q_values_all.gather(1, batch_action)       # [batch_size, 1]

        # 2) Double DQN logic:
        #    - Use policy_net to pick best action in the next state
        #    - Then evaluate that action with target_net
        with torch.no_grad():
            # a) Next state Q-values from policy_net (for action selection)
            next_q_policy = self.policy_net(batch_next_state)  # [batch_size, action_dim]

            # b) Next state Q-values from target_net (for value)
            next_q_target = self.target_net(batch_next_state)  # [batch_size, action_dim]

            # We'll store the Double DQN chosen action's Q-value in next_q_values
            next_q_values = torch.zeros((self.batch_size, 1), device=self.device)

            for i, valid_acts in enumerate(valid_actions_next):
                if len(valid_acts) > 0:
                    # i) pick best valid action from policy_net
                    best_action_idx = torch.argmax(next_q_policy[i, valid_acts])
                    best_action = valid_acts[best_action_idx]  # This is the actual action index
                    
                    # ii) evaluate Q-value for that action from target_net
                    next_q_values[i] = next_q_target[i, best_action]
                else:
                    # If no valid actions, Q-value remains 0
                    next_q_values[i] = 0.0

        # 3) Bellman target:
        #    target = reward + (1 - done) * gamma * Q_target(next_state, best_action)
        #    where best_action is chosen by the policy_net
        expected_q = batch_reward + (1 - batch_done) * self.gamma * next_q_values

        # 4) Compute loss and update
        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
