import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from PER import PrioritizedReplayMemory

class DQNet(nn.Module):
    def __init__(self, state_dim=45, action_dim=44):
        super(DQNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.010, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=100, rng=None,
                 alpha=0.6,        
                 beta_start=0.4,    
                 beta_increment=1e-5, CUDA=True): 
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        # PER 관련
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        #self.memory = deque(maxlen=buffer_size)
        # Replay memory를 PER로 교체
        self.memory = PrioritizedReplayMemory(buffer_size, alpha=self.alpha, beta=self.beta)
        
        #Cuda device
        if CUDA:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        
        self.policy_net = DQNet(state_dim, action_dim).to(self.device)
        self.target_net = DQNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr = lr, weight_decay=1e-5)
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
                if action not in valid_actions:
                    print(f"Invalid actions are selected. valid actions : {valid_actions}, action : {action}")
            
            return action


    def push_memory(self, transition):
        # Transition: (state, action, reward, next_state, done, valid_actions_next)
        self.memory.push(transition)
    
    def optimize_model(self):
        # 메모리에 쌓인 샘플이 batch_size보다 적으면 학습 불가
        if len(self.memory) < self.batch_size:
            return
        
        # PER의 sample: transitions, indices, IS weights 반환
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        # transitions = list of tuples
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        next_states = np.array([t.next_state for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        valid_actions_next = [t.valid_actions_next for t in transitions]
        
        batch_state      = torch.from_numpy(states).to(self.device)
        batch_action     = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        batch_reward     = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        batch_next_state = torch.from_numpy(next_states).to(self.device)
        batch_done       = torch.from_numpy(dones).unsqueeze(1).to(self.device)
        weights          = weights.to(self.device)  # [batch_size, 1] 형태
        
        # 1) 현재 Q값 (Policy net)
        q_values_all = self.policy_net(batch_state)            # [batch_size, action_dim]
        current_q = q_values_all.gather(1, batch_action)       # [batch_size, 1]
        
        # 2) Double DQN: next action은 policy_net에서 고르고, Q값은 target_net에서 취함
        with torch.no_grad():
            next_q_policy = self.policy_net(batch_next_state)  # [batch_size, action_dim]
            next_q_target = self.target_net(batch_next_state)  # [batch_size, action_dim]
            
            next_q_values = torch.zeros((self.batch_size, 1), device=self.device)
            for i, valid_acts in enumerate(valid_actions_next):
                if len(valid_acts) > 0:
                    best_action_idx = torch.argmax(next_q_policy[i, valid_acts])
                    best_action = valid_acts[best_action_idx]
                    next_q_values[i] = next_q_target[i, best_action]
                else:
                    next_q_values[i] = 0.0
        
        # 3) Bellman Target
        expected_q = batch_reward + (1 - batch_done) * self.gamma * next_q_values
        
        # TD Error(오차) = expected_q - current_q
        td_error = expected_q - current_q
        
        # 4) PER에서 priority로 사용할 값. (HuberLoss 등으로 해도 됨)
        # 우선순위는 TD 오차의 절댓값으로. detach()로 그래프에서 분리
        new_priorities = td_error.detach().abs().cpu().numpy().flatten()
        
        # 5) Loss = IS weight * MSE
        # 보통 (td_error^2)에 IS weight를 곱하고 평균 내는 방식
        loss = (weights * (td_error ** 2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping (옵션)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # PER 우선순위 갱신
        self.memory.update_priorities(indices, new_priorities)

        # beta를 조금씩 증가(학습 후반 갈수록 학습 안정을 위해)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
