from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done', 'valid_actions_next'))

class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay Buffer
    - capacity: 버퍼 최대 크기
    - alpha: TD 오차를 우선순위로 반영하는 정도(0이면 uniform, 1이면 오차 그대로 반영)
    - beta: IS(Importance Sampling) 보정 계수
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.min_priority = 1e-6  # 우선순위가 0이 되지 않도록 하는 작은 값

    def push(self, transition):
        """
        transition: (state, action, reward, next_state, done, valid_actions_next)
        """
        max_priority = np.percentile(self.priorities, 90) if self.priorities else 1.0
        transition = Transition(*transition)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, current_epsilon):
        """
        PER-based sampling with epsilon-dependent behavior.
        """
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])

        # Mix PER with uniform sampling based on the latest epsilon
        mix_ratio = max(0.1, 1 - current_epsilon)  # More PER as epsilon decreases
        per_size = int(batch_size * mix_ratio)
        uniform_size = batch_size - per_size

        # Ensure we don't sample more than available elements
        available_samples = len(self.buffer)

        # Uniform sampling
        uniform_indices = np.random.choice(
            available_samples, min(uniform_size, available_samples), replace=False
        )

        # PER sampling
        if per_size > 0:
            probs = priorities ** self.alpha
            probs /= probs.sum()

            per_indices = np.random.choice(
                available_samples, min(per_size, available_samples), p=probs, replace=False
            )

            weights = (available_samples * probs[per_indices]) ** (-self.beta)
            weights /= weights.max()
        else:
            per_indices = np.array([])  # Empty array if PER not used
            weights = np.ones(uniform_size)  # Default weight of 1 for uniform samples

        # If batch is too small, add extra uniform samples
        total_samples = len(per_indices) + len(uniform_indices)
        if total_samples < batch_size:
            extra_needed = batch_size - total_samples
            extra_indices = np.random.choice(available_samples, extra_needed, replace=True)
            uniform_indices = np.concatenate((uniform_indices, extra_indices))

        #  Use np.concatenate to merge indices efficiently
        final_indices = np.concatenate((per_indices, uniform_indices)).astype(int)
        final_transitions = [self.buffer[idx] for idx in final_indices]
        final_weights = torch.FloatTensor(np.concatenate((weights, np.ones(len(uniform_indices))))).unsqueeze(1)

        return final_transitions, final_indices, final_weights

    def update_priorities(self, indices, priorities):
        """
        새로 계산된 TD 오차(priority)로 우선순위를 갱신
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(abs(prio), self.min_priority)

    def __len__(self):
        return len(self.buffer)