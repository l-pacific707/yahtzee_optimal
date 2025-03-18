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
        # 처음에 최대 우선순위를 부여해 새로 들어온 샘플이 무조건 샘플될 기회를 준다.
        # 혹은 버퍼 내 가장 큰 priority를 가져와 부여하는 식도 가능함
        max_priority = max(self.priorities) if self.priorities else 1.0
        transition = Transition(*transition)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        PER에 따라 batch_size만큼 샘플링하여 반환:
        - transitions: 샘플된 transition
        - indices: 샘플된 인덱스 목록
        - weights: IS(Importance Sampling) weight 목록
        """
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        
        # 확률 분포: p_i = priority_i^alpha / sum(priority^alpha)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # probs 기반으로 인덱스 샘플링
        indices = np.random.choice(len(priorities), batch_size, p=probs, replace=False)

        # IS weight 계산
        # w_i = ( N * P(i) )^-beta
        # 여기서 P(i)=probs[i], N=len(priorities)
        weights = (len(priorities) * probs[indices]) ** (-self.beta)
        # 가중치 정규화
        weights /= weights.max()
        
        transitions = [self.buffer[idx] for idx in indices]
        return transitions, indices, torch.FloatTensor(weights).unsqueeze(1)

    def update_priorities(self, indices, priorities):
        """
        새로 계산된 TD 오차(priority)로 우선순위를 갱신
        """
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = max(abs(prio), self.min_priority)

    def __len__(self):
        return len(self.buffer)