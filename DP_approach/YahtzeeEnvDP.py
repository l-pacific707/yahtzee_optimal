import numpy as np
from itertools import combinations_with_replacement
from itertools import combinations
from itertools import product
from collections import defaultdict
import math
import pickle



class YahtzeeEnvDP:
    NUMBER_OF_DICE = 5
    NUMBER_OF_SIDES = 6
    LENGTH_OF_CATEGORY = 12

    ALL_DICE_STATE = np.array(list(combinations_with_replacement(range(1, 1 + NUMBER_OF_SIDES), NUMBER_OF_DICE)))

    @classmethod
    def get_all_state(cls):
        """Generate all possible states as a proper 2D array (each row is a full state vector)."""

        all_dice_state = cls.ALL_DICE_STATE  # Shape: (252, 5)
        all_availables = np.array(list(product([0, 1], repeat=cls.LENGTH_OF_CATEGORY)))  # Shape: (4096, 12)        
        all_rerolls = np.array([0, 1, 2, 3])  # Shape: (4,)
        all_upscores = np.array(range(0, 64))  # Shape: (64,)

        print(f"all_dice_state shape: {all_dice_state.shape}")  # Should be (252, 5)
        print(f"all_availables shape: {all_availables.shape}")  # Should be (?, 12)
        print(f"all_rerolls shape: {all_rerolls.shape}")  # Should be (4,)
        print(f"all_upscores shape: {all_upscores.shape}")  # Should be (64,)

        # Generate Cartesian product
        state_list = []
        steps = 0
        for dice, available, upscore, reroll in product(all_dice_state, all_availables, all_upscores, all_rerolls):
            state_vector = np.hstack([dice, available, [upscore], [reroll]])  # Ensure all elements are in 1D form
            state_list.append(state_vector)
            steps += 1
            if steps % 1000000 == 0:
                print(f'steps {steps} is done.')

        # Convert to a 2D NumPy array
        all_state = np.array(state_list, dtype=np.int32)  # Shape: (total_states, 19)

        print(f"Final state space shape: {all_state.shape}")  # Should be (N, 19)
        cls.save_object(all_state, "state_space.pkl")
        return all_state
    
    @classmethod
    def initialize_class(cls, load_filepath = "state_space.pkl"):
        """Initialize class variables after definition."""
        try:
            cls.ALL_STATE = cls.load_object(load_filepath)
        except FileNotFoundError:
            cls.ALL_STATE = cls.get_all_state()
    
    def __init__(self, dice_state = np.zeros(NUMBER_OF_DICE, dtype = np.int32), availables = np.zeros(LENGTH_OF_CATEGORY , dtype=np.int32), upscore = 0, reroll = 0) -> None:
        self.dice_state = dice_state
        self.availables = availables # total 12 categories
        self.upscore = upscore # 0-63 
        self.reroll = reroll # 0-3
        
    
    def reset(self):
        self.dice = np.zeros(5, dtype = np.int32)
        self.availables = np.zeros(12 , dtype=np.int32) # total 12 categories
        self.upscore = 0 # 0-63 
        self.reroll = 0 # 0-3
        self.state = self.get_state()
    
    @staticmethod
    def transition_prob(state, action, next_state) -> float:
        """calculate transition probability from s to s'

        Args:
            state (ndarray): (dicestate(5), availables(12, ), upscore(1), reroll(1)) : shape(19,)
            action (integer): initiate roll (0), reroll (1-31), scoring(32-43)
            next_state (ndarray): changed state, (dicestate(5), availables(12), upscore(1), reroll(1)) : shape(19,)
        """
        current_dice = state[0:5]
        current_availables = state[5:17]
        current_upscore = state[17]
        current_reroll = state[18]
        
        next_dice = next_state[0:5]
        next_availables = next_state[5:17]
        next_upscore = next_state[17]
        next_reroll = next_state[18]

        if 1<=action<=31 :
            #Reroll
            if next_reroll != current_reroll - 1:
                return 0.0
            elif np.any(next_availables != current_availables):
                return 0.0
            elif next_upscore != current_upscore:
                return 0.0
            else:
                reroll_mask = YahtzeeEnvDP.int_to_bitmask(action)
                non_selected_nums = []
                for i, yes in enumerate(reroll_mask):
                    if not yes:
                        non_selected_nums.append(current_dice[i])
                for nums in non_selected_nums:
                    if nums not in next_dice:
                        return 0.0
                    else:
                        next_dice.remove(nums)
                k = YahtzeeEnvDP.NUMBER_OF_DICE - len(next_dice) # how many dice will be reroll-ed?
                
                diffs = defaultdict(int) # num : occurences dictionary
                
                for i, num in np.sort(next_dice):
                    diffs[num] += 1
                
                prob = ((1/6) ** k ) * math.factorial(k) # default probability
                for num in diffs.keys():
                    prob /= math.factorial(diffs[num])
                return prob
        elif 32<=action<=43:
            #Scoring
            category_tobe_scored_idx = action - 32 # 0-11  (index)
            change = (current_availables != next_availables)
            if np.sum(change) == 1 and change[category_tobe_scored_idx] == True and next_reroll == 3 and current_availables[category_tobe_scored_idx] == 1 and next_dice == np.zeros(YahtzeeEnvDP.LENGTH_OF_CATEGORY, dtype = np.int32) and current_upscore == next_upscore:
                return 1.0
            else:
                return 0.0
        elif action == 0:
            #Initiate roll
            if current_reroll != 3 or next_reroll !=2 :
                return 0.0
            elif np.any(next_availables != current_availables):
                return 0.0
            elif next_upscore != current_upscore:
                return 0.0
            else:
                k = YahtzeeEnvDP.NUMBER_OF_DICE # how many dice will be reroll-ed?
                
                diffs = defaultdict(int) # num : occurences dictionary
                
                for i, num in np.sort(next_dice):
                    diffs[num] += 1
                
                prob = ((1/6) ** k ) * math.factorial(k) # default probability
                for num in diffs.keys():
                    prob /= math.factorial(diffs[num])
                return prob

    def make_transition_prob_table(state_space, save_filepath = "transition_prob_table.pkl"):
        """Make dictionary of transition probability table


        Args:
            state_space (_type_): _description_
            action_space (_type_): _description_

        Returns:
            dictionary
            -keys : (current state, action, next_state)
            - values : non-zero transition probability , i.e. p(s_|s,a)
        """
        table = defaultdict()
        for s in state_space:
            valid_action = YahtzeeEnvDP(s[0:5], s[5:17], s[17], s[18]).get_valid_action()
            for a in valid_action:
                for s_ in state_space:
                    prob = YahtzeeEnvDP.transition_prob(s,a,s_)
                    if prob != 0:
                        table[(s,a,s_)] = prob
        YahtzeeEnvDP.save_object(table, save_filepath)
        return table
    
    def get_state(self) -> np.ndarray:
        """Return state

        Returns:
            ndarray : state, shape: (19,)
        """
        return np.concatenate(self.dice_state, self.availables, np.array([self.upscore]), np.array([self.reroll]))

    def get_valid_action(self) :
        state = self.get_state()
        current_availables = state[5:17]
        current_reroll = state[18]
        
        valid_actions = []
        if current_reroll == 3:
            valid_actions.append(0)
            return valid_actions
        #reroll options
        elif current_reroll >= 1:
            for i in range(1,32):
                valid_actions.append(i)
        #scoring options
        for i, num in enumerate(current_availables):
            if num == 1:
                valid_actions.append(i+32)
        return sorted(valid_actions)
        
        
    @staticmethod
    def policy_evaluation(value, transition_prob, state_space, reward_func, policy, gamma = 1.0, n_steps = 1, initialize=True):
        """Evaluate the value function for current policy.

        Args:
            value (dict) : value function for state s
            transition_prob (_type_): 
            state_space (_type_): _description_
            reward_func (_type_): _description_
            policy (_type_): _description_
            gamma (_type_): _description_
            n_steps (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """        
        if initialize:
            #initialize V(s)
            V = defaultdict()
            for s in state_space:
                V[s] = 0
        else:
            V = value
        # iteration start
        for _ in range(n_steps):
            for s in state_space:
                v = V[s]
                #assuming deterministic policy
                for s_ in state_space:
                    V[s] = transition_prob(s, policy[s], s_) * (reward_func(s,policy[s]) + gamma * V[s_])
        return V
    
    @staticmethod
    def policy_improvement(value, transition_prob, state_space, reward_func, policy : dict, gamma : float = 1.0):
        
        # greedy select action policy
        V = value
        policy_stable = True
        for s in state_space:
            old_action = policy[s]
            current_dice = s[0:5]
            current_availables = s[5:17]
            current_upscore = s[17]
            current_reroll = s[18]
            env = YahtzeeEnvDP(current_dice, current_availables, current_upscore, current_reroll)
            valid_action = env.get_valid_action()
            action_choice = []
            for a in valid_action:
                temp = 0
                for s_ in state_space:
                    temp += transition_prob(s, a, s_) * (reward_func(s,a) + gamma * V[s_])
                action_choice.append(temp)
            policy[s] = valid_action[np.argmax(action_choice)]
            if old_action != policy[s]:
                policy_stable = False
        return policy, policy_stable
    
    @staticmethod
    def int_to_bitmask(num):
        """Change an integer number to a 5-bit mask corresponding to num in binary representation.

        Input: integer number (1-31)
        Output: list of integers representing a 5-bit mask.

        Example:
            input: 21
            return: [1, 0, 1, 1, 0] 
        """
        if not 1 <= num <= 31:
            raise ValueError("Input number must be between 0(inclusive) and 30 (inclusive).")

        # Convert to binary, remove '0b' prefix, and fill with leading zeros to ensure 5 bits
        bitmask = list(map(int, format(num, '05b'))) # change the num+1 to num

        return bitmask
    
    @staticmethod
    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load_object(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    

