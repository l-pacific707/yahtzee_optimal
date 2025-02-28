#!/usr/bin/env python
"""
Reinforcement Learning for the Yacht Game using DQN

The Yacht game is similar to Yahtzee: in each of 12 turns you roll five dice up to 3 times.
You then select one of 12 scoring categories (six “upper” ones plus six “lower” ones).
If the sum of the scores in the upper six categories is at least 63,
a bonus of 35 points is awarded.

In this implementation:
 - The state is encoded as:
    • normalized turn number (turn/12)
    • normalized roll number (roll/2)  [roll 0,1,2]
    • the five dice as one–hot vectors (5 × 6 = 30 numbers)
    • a binary 12–dim vector marking which categories remain available.
   (Total state vector length = 44.)
 - The (fixed) action space has 44 actions:
    • Actions 0–31: “re–roll” actions (each interpreted as a 5–bit mask for which dice to re–roll).
      (Re–roll actions are available only if you haven’t yet used all 3 rolls.)
    • Actions 32–43: “choose category” actions (action–32 corresponds to category 0..11).
 - The environment (YachtEnv) handles the dice rolls, category scoring (using standard rules),
   and at game end it adds a bonus of 35 if the sum of upper–section scores is ≥63.
 - A DQN agent (with experience replay and a target network) is used to choose among valid actions.
   (Invalid actions are masked out when selecting the best Q–value.)
 
Run this script to train the agent and then see one test game printed to the console.
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Scoring function for each category.
# We assume the following category mapping (indices 0-11):
# 0: Ones, 1: Twos, 2: Threes, 3: Fours, 4: Fives, 5: Sixes  (upper section)
# 6: Choice (sum of dice)
# 7: Four-of-a-Kind (if at least four dice are the same, score sum of dice)
# 8: Full House (if three of one number and two of another, score 15 points)
# 9: Little Straight (dice = 1,2,3,4,5 in any order, score 30 points)
# 10: Big Straight (dice = 2,3,4,5,6 in any order, score 30 points)
# 11: Yacht (all dice identical, score 50 points)
# ---------------------------
def score_category(dice, category):
    # dice: list of 5 integers (each 1..6)
    if category == 0:  # Ones
        return dice.count(1) * 1
    elif category == 1:  # Twos
        return dice.count(2) * 2
    elif category == 2:  # Threes
        return dice.count(3) * 3
    elif category == 3:  # Fours
        return dice.count(4) * 4
    elif category == 4:  # Fives
        return dice.count(5) * 5
    elif category == 5:  # Sixes
        return dice.count(6) * 6
    elif category == 6:  # Choice
        return sum(dice)
    elif category == 7:  # Four-of-a-Kind: if any number appears at least 4 times
        for i in range(1,7):
            if dice.count(i) >= 4:
                return sum(dice)
        return 0
    elif category == 8:  # Full House: three of one number and two of another
        counts = [dice.count(i) for i in range(1,7)]
        sorted_counts = sorted(counts)
        if sorted_counts == [0,0,0,2,3,0] or (2 in counts and 3 in counts):
            return 15
        return 0
    elif category == 9:  # Little Straight: 1-2-3-4-5
        if sorted(dice) == [1,2,3,4,5]:
            return 30
        return 0
    elif category == 10:  # Big Straight: 2-3-4-5-6
        if sorted(dice) == [2,3,4,5,6]:
            return 30
        return 0
    elif category == 11:  # Yacht: all dice the same
        if len(set(dice)) == 1:
            return 50
        return 0
    else:
        return 0

# ---------------------------
# Yacht Environment
# ---------------------------
class YachtEnv:
    def __init__(self):
        # There are 12 categories; True indicates the category is still available.
        self.available_categories = [True] * 12
        self.upper_categories = set(range(6))  # categories 0-5
        self.reset()

    def reset(self):
        self.turn = 1            # Turn count from 1 to 12.
        self.total_score = 0
        self.upper_score = 0     # Sum of scores in the upper section.
        self.available_categories = [True] * 12
        self.roll = 0            # Roll count within the turn (0,1,2)
        # Initial roll: roll all 5 dice.
        self.dice = [random.randint(1,6) for _ in range(5)]
        self.done = False
        return self.get_state()

    def get_state(self):
        """Return the current state as a numpy array (length 44)."""
        # Normalize turn and roll numbers.
        turn_norm = self.turn / 12.0
        roll_norm = self.roll / 2.0
        # Encode dice as 5 one-hot vectors (each of length 6).
        dice_encoding = []
        for d in self.dice:
            one_hot = [0] * 6
            one_hot[d - 1] = 1
            dice_encoding.extend(one_hot)
        # Encode available categories as 12 binary values.
        avail_encoding = [1 if avail else 0 for avail in self.available_categories]
        state = np.array([turn_norm, roll_norm] + dice_encoding + avail_encoding, dtype=np.float32)
        return state

    def get_valid_actions(self):
        """
        Returns a list of valid actions (indices 0..43).
        - If the current roll is less than 2, all 32 re-roll actions (0-31) are allowed.
        - At any time, you may choose a category (actions 32-43) if that category is available.
        """
        valid = []
        if self.roll < 2:
            valid.extend(list(range(32)))  # Re-roll actions.
        for cat in range(12):
            if self.available_categories[cat]:
                valid.append(32 + cat)  # Category selection actions.
        return valid

    def step(self, action):
        """
        Take an action in the environment.
        Actions 0-31: re-roll decision. Interpret the action as a 5–bit mask;
            if bit i is 1 then re-roll the i–th die.
        Actions 32–43: choose the scoring category (action-32).
        Returns: (next_state, reward, done, info)
        """
        valid = self.get_valid_actions()
        if action not in valid:
            # If an invalid action is selected, penalize and return the same state.
            reward = -10
            next_state = self.get_state()
            return next_state, reward, self.done, {}

        # --- Re-roll Action ---
        if action < 32:
            # Re-roll action: convert the integer (0-31) to a 5-bit mask.
            mask = [(action >> i) & 1 for i in range(5)]
            # For each die with mask 1, re-roll it.
            for i in range(5):
                if mask[i] == 1:
                    self.dice[i] = random.randint(1,6)
            self.roll += 1
            reward = 0
            next_state = self.get_state()
            return next_state, reward, self.done, {}

        # --- Category Selection Action ---
        else:
            cat = action - 32
            # Score the current dice for the selected category.
            score = score_category(self.dice, cat)
            self.total_score += score
            if cat in self.upper_categories:
                self.upper_score += score
            # Mark the category as used.
            self.available_categories[cat] = False
            reward = score  # Reward for this turn.

            if self.turn == 12:
                # End of game: add bonus if upper section >= 63.
                if self.upper_score >= 63:
                    reward += 35
                    self.total_score += 35
                self.done = True
                next_state = self.get_state()
                return next_state, reward, self.done, {}
            else:
                # Prepare for the next turn.
                self.turn += 1
                self.roll = 0
                self.dice = [random.randint(1,6) for _ in range(5)]
                next_state = self.get_state()
                return next_state, reward, self.done, {}

# ---------------------------
# Deep Q-Network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ---------------------------
# DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.995,
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
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def select_action(self, state, valid_actions):
        """
        Choose an action using an epsilon-greedy strategy,
        but only among the valid actions provided.
        """
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor).squeeze(0).detach().numpy()
                # Mask out invalid actions by assigning -infinity.
                masked_q = np.full(self.action_dim, -np.inf)
                for a in valid_actions:
                    masked_q[a] = q_values[a]
                action = int(np.argmax(masked_q))
                return action

    def push_memory(self, transition):
        # Transition: (state, action, reward, next_state, done, valid_actions_next)
        self.memory.append(transition)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        # Unpack transitions.
        batch_state = torch.FloatTensor([t[0] for t in transitions])
        batch_action = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1)
        batch_reward = torch.FloatTensor([t[2] for t in transitions]).unsqueeze(1)
        batch_next_state = torch.FloatTensor([t[3] for t in transitions])
        batch_done = torch.FloatTensor([float(t[4]) for t in transitions]).unsqueeze(1)
        
        # Current Q values for the actions taken.
        current_q = self.policy_net(batch_state).gather(1, batch_action)
        
        # Compute next Q values from the target network.
        next_q_all = self.target_net(batch_next_state).detach().numpy()  # (batch, action_dim)
        max_next_q = []
        for i, t in enumerate(transitions):
            valid_next = t[5]
            if t[4]:  # if done then no next Q value
                max_next_q.append(0.0)
            else:
                q_vals = next_q_all[i]
                # Only consider valid actions in the next state.
                masked = [q_vals[a] for a in valid_next]
                max_next_q.append(max(masked) if masked else 0.0)
        max_next_q = torch.FloatTensor(max_next_q).unsqueeze(1)
        expected_q = batch_reward + self.gamma * max_next_q
        
        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon gradually.
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------------------
# Training Loop
# ---------------------------
def train_agent(num_episodes=1000):
    env = YachtEnv()
    state_dim = 44   # see get_state() above.
    action_dim = 44  # fixed action space (32 re-roll + 12 categories)
    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []

    for i_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # For the next state, record its valid actions (empty if done).
            valid_actions_next = env.get_valid_actions() if not done else []
            agent.push_memory((state, action, reward, next_state, done, valid_actions_next))
            state = next_state
            agent.optimize_model()
            if done:
                break

        if i_episode % agent.target_update == 0:
            agent.update_target()
        episode_rewards.append(total_reward)
        if (i_episode + 1) % 100 == 0:
            print(f"Episode {i_episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    return agent, episode_rewards

# ---------------------------
# Main: Train and test the agent.
# ---------------------------
if __name__ == "__main__":
    # Train the agent (adjust num_episodes as desired)
    trained_agent, rewards = train_agent(num_episodes=3000)

    # Test the learned policy on one game.
    print("\n--- Testing learned policy ---")
    env = YachtEnv()
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        valid_actions = env.get_valid_actions()
        action = trained_agent.select_action(state, valid_actions)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        # For clarity, print the action and current dice.
        if action < 32:
            # Re-roll action: show the 5-bit mask.
            mask = [(action >> i) & 1 for i in range(5)]
            action_desc = f"Re-roll mask {mask}"
        else:
            cat = action - 32
            action_desc = f"Select category {cat}"
        print(f"Turn {env.turn if not done else 12} | Roll {env.roll} | Action: {action_desc} | Reward: {reward} | Dice: {env.dice}")

    print("Test game total score:", total_reward)
