import gymnasium as gym
import numpy as np
import math


class YahtzeeEnv(gym.Env):
    def __init__(self):
        super(YahtzeeEnv, self).__init__()
        # Define spaces ( )
        self.observation_space = gym.spaces.Box(low=0, high=50, shape=(45,), dtype=np.int32)  # ((dice_result: 5 * 6 bit for one hot encoding), reroll, turn, [scores: 12], bonus )
        self.action_space = gym.spaces.Discrete(44)  # 0 : initial roll, 1-31: reroll, 32-43: filling the score)
        self._reset()
        self.rng = np.random.default_rng()

    def _reset(self):
        # Reset environment state (dice, categories, etc.)
        # Game states
        self.dice = np.zeros((5, 6), dtype=np.int32)  # one hot vectors in array
        self.rerolls = 3
        self.turn = 0
        self.scorecard = np.zeros(12, dtype=np.int32)
        self.bonus = False
        self.done = False
        self.bonusRewarded = False
        self.scored = np.full((12,), False, dtype=np.bool)

        # Return initial observation -> do we need this?

    def _reroll_under_mask(self, mask: list):
        "Reroll dice result under the bitmask"
        if self.rerolls == 0:
<<<<<<< Updated upstream
            raise ValueError(f"No reroll remains. ")
=======
            raise ValueError("No reroll remains.")
>>>>>>> Stashed changes
        else:
            self.rerolls -= 1
            for i , _ in enumerate(mask):
                if mask[i] == 1:
                    self.dice[i, :] = np.zeros((1, 6), dtype=np.int32)  # reset ith die value
                    j = self.rng.integers(low=0, high=5, endpoint=True, dtype=np.int32)
                    self.dice[i, j] = 1

    def get_score_for_action(self, action) -> int:
        """ self.dice: 2D numpy array (5*6), each row represents one number under one-hot encoding
            action : 31-42 integer number
            Return : score(int) for selected action
        """
        scoreto = action - 32  # Changed from 31 to 32
        numbers = [0, 0, 0, 0, 0, 0]  # how many occurences are there for 1,2,...,5,6?
        meresum = 0
        for i in range(5):
            if self.dice[i][0] == 1:  ## ith die is 1
                numbers[0] += 1
            elif self.dice[i][1] == 1:  ## ith die is 2
                numbers[1] += 1
            elif self.dice[i][2] == 1:
                numbers[2] += 1
            elif self.dice[i][3] == 1:
                numbers[3] += 1
            elif self.dice[i][4] == 1:
                numbers[4] += 1
            elif self.dice[i][5] == 1:
                numbers[5] += 1
            else:
                continue
        for i in range(5):
            meresum += numbers[i] * (i + 1)

        if scoreto == 0:  # Ones
            return numbers[0] * 1
        elif scoreto == 1:  # Twos
            return numbers[1] * 2
        elif scoreto == 2:  # Threes
            return numbers[2] * 3
        elif scoreto == 3:  # Fours
            return numbers[3] * 4
        elif scoreto == 4:  # Fives
            return numbers[4] * 5
        elif scoreto == 5:  # Sixes
            return numbers[5] * 6
        elif scoreto == 6:  # Choice
            return meresum
        elif scoreto == 7:  # Four-of-a-Kind: if any number appears at least 4 times
            for i in range(5):
                if numbers[i] >= 4:
                    return meresum
                else:
                    continue
            return 0
        elif scoreto == 8:  # Full House: three of one number and two of another
            return meresum if (3 in numbers) and (2 in numbers) else 0
        elif scoreto == 9:  # Little Straight: 1,2,3,4; 2,3,4,5; 3,4,5,6

            if numbers[0:4] == [1, 1, 1, 1]:
                return 15
            elif numbers[1:5] == [1, 1, 1, 1]:
                return 15
            elif numbers[2:] == [1, 1, 1, 1]:
                return 15
            else:
                return 0
        elif scoreto == 10:  # Big Straight: 2-3-4-5-6; 1,2,3,4,5
            if (numbers[:5] == [1, 1, 1, 1, 1]) or (numbers[1:6] == [1, 1, 1, 1, 1]):
                return 30
            else:
                return 0
        elif scoreto == 11:  # Yacht: all dice the same
            try:
                numbers.index(5)
                return 50
            except ValueError:
                return 0
        else:
            print("Undealt case raised. scoring function must be modified.")
            return 0

    def _score_action(self, action, score=None):
        """Fill the score in scorecard with selected action. (in-place)
         This function does
         1) fill the score
         2) reset rerolls to 3
         3) turn increases
         4) check game end
         5) check bonus point is possible


        Args:
            action (int): 32-43 integer.
            score (int, optional) : score for that category
        No return value
        """
        if action < 32 or action > 43:
            raise ValueError(f"Invalid scoring action: {action}. Action must be between 32 and 43 for scoring.")

        index = action - 32  # Changed from 31 to 32
        if self.scored[index]:
            raise ValueError(f"Already filled in that category : {index}, value : {self.scorecard[index]}")
        val = score if score is not None else self.get_score_for_action(action)
        self.scorecard[index] = val
        self.scored[index] = True
        self.rerolls = 3
        self.turn += 1

        # check if game is completed
        if self.turn == 12:  # game end
            self.done = True
        # check the bonus point is possible
        if np.sum(self.scorecard[0:6]) >= 63:
            self.bonus = True

    def _initiate_turn(self):
        if self.rerolls == 3:
            self._reroll_under_mask([1, 1, 1, 1, 1])

    def get_state(self) -> np.ndarray:
        return np.concatenate([
            self.dice.flatten(),
            np.array([self.rerolls]),  # Convert scalar to array
            np.array([self.turn]),  # Convert scalar to array
            self.scorecard,
            np.array([self.bonus], dtype=int)  # Convert boolean to int
        ])

    def get_valid_action(self) -> list:
        """Return list of valid action, e.g.)[1,4,5,43] action: 0(initiate roll), 1-31(reroll), 32-43(scoring); integer """

        if self.rerolls == 3:
            return [0] #only initial roll
        elif self.rerolls == 0:
            # only scoring option is available
            valids = []
            for i, filled in enumerate(self.scored):
                if not filled:
                    valids.append(i + 32)  # Changed from 31 to 32
            return valids
        else:
            # We have rerolls, and scoring is available as well
            valids = list(range(1, 32))  # Changed from 0 to 1
            for i, filled in enumerate(self.scored):
                if not filled:
                    valids.append(i + 32)  # Changed from 31 to 32
            return valids

    def step(self, action):
        # 1) Apply action (roll dice or choose category, etc.)
        # 2) Calculate reward
        # 3) Determine if episode is done
        # 4) Return the next state, reward, done, and optionally info dict

        valid = self.get_valid_action()
        # print(f"***Valid actions in step function(output of get_valid_function) : {valid}***")

        if action not in valid:
            # If an invalid action is selected, penalize and return the same state.
            reward = -15
            next_state = self.get_state()
            return next_state, reward, self.done, {}

        ## turn initiation
        if action == 0:
            reward = 0
            self._initiate_turn()
            next_state = self.get_state()
            return next_state, reward, self.done, {}

        ## reroll action
        if 1 <= action <= 31:  # Changed from < 31 to range 1-31
            mask = self.int_to_bitmask(action)
            self._reroll_under_mask(mask)
<<<<<<< Updated upstream
            reward = self.get_sum_possible_score() * 0.08  # since this is not actual reward (not finally scored value)
=======
            if self.rerolls == 0 :
                try:
                    # averaging over 12 categories
                    reward = self.get_sum_possible_score() / len(valid)
                except ZeroDivisionError:
                    reward = 0
            else :
                reward = self.get_expected_reward()
>>>>>>> Stashed changes
            next_state = self.get_state()
            return next_state, reward, self.done, {}

        ## scoring action
        elif 32 <= action <= 43:  # Changed to range 32-43
            score = self.get_score_for_action(action)
            reward = score 
            self._score_action(action, score)
            if self.bonus and (self.bonusRewarded is False):
                reward += 35
                self.bonusRewarded = True
            # reward += score / 63  # bonus contribution
            next_state = self.get_state()
            return next_state, reward, self.done, {}
        else:
            raise ValueError(f"Invalid action: {action}. Action must be between 0 and 43.")

    def get_sum_possible_score(self) -> int:
        valids = self.get_valid_action()
        sum = 0
        for sAction in valids:
            if sAction >= 32:
                sum += self.get_score_for_action(sAction)
        return sum

<<<<<<< Updated upstream
=======
    def get_expected_reward(self) -> float:
        """
        Computes the expected reward for valid scoring actions.
        All actions follow the same probability logic.
        """
        valids = self.get_valid_action()
        temp = [32,33,34,35,36,37]
        valids = [x for x in valids if x in temp]
        reward = 0
        def _expected_score(category_value: int) -> float:
            """Calculates the expected score for any scoring category (ones to sixes)."""
            count = sum(1 for i in range(5) if self.dice[i][category_value - 1] == 1)  # Count matching dice
            diffs = 5 - count  # Number of dice that need to change
            expected_reward = 0
            for k in range(diffs + 1):  # Include case where no dice change
                expected_reward += (count + k) * category_value * math.comb(diffs, k) * (1 / (6 ** k)) * ((5 / 6) ** (diffs - k))
            return expected_reward

        for action in valids:
            # Convert action number to category value (32→1, 33→2, ..., 37→6)
            reward += _expected_score(action - 31)  
        return reward / len(valids) if valids else 0  # Prevent division by zero
    

>>>>>>> Stashed changes
    @staticmethod
    def int_to_bitmask(num):
        """Change an integer number to a 5-bit mask corresponding to (num+1) in binary representation.

        Input: integer number (1-31)
        Output: list of integers representing a 5-bit mask.

        Example:
            input: 21
            return: [1, 0, 1, 1, 0] 
        """
<<<<<<< Updated upstream
        if not (1 <= num <= 31):
=======
        if not 1 <= num <= 31:
>>>>>>> Stashed changes
            raise ValueError("Input number must be between 0(inclusive) and 30 (inclusive).")

        # Convert to binary, remove '0b' prefix, and fill with leading zeros to ensure 5 bits
        bitmask = list(map(int, format(num, '05b'))) # change the num+1 to num

        return bitmask
