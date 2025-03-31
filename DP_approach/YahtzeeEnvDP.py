import numpy as np
from collections import defaultdict
from collections import Counter
from itertools import combinations_with_replacement
import logging
import pickle
import logger_setup
import math


logger = logging.getLogger(__name__)


def show_all_attributes(obj):
    for attr in dir(obj):
        if not attr.startswith("__"):
            print(f"{attr} = {getattr(obj, attr)}")

class Roll:
    NUMBER_OF_DICE = 5
    NUMBER_OF_SIDES = 6
    ALL_DICE_STATES = ((0,0,0,0,0), *tuple(combinations_with_replacement(range(1, NUMBER_OF_SIDES + 1), NUMBER_OF_DICE)))
    
    def __init__(self, roll= None, seed = None):
        self.rng = np.random.default_rng(seed)
        if roll is None:
            self.r = np.zeros(self.NUMBER_OF_DICE, dtype = np.int32)
        else:
            self.r = roll
        
    def reset(self):
        self.r = np.zeros(self.NUMBER_OF_DICE, dtype = np.int32)
    
    def reroll(self, keep_mask = 0b00000) -> None:
        """reroll current roll under keep_mask .
        - 0b0001 means first four number will be changed.
        - this method changes instance's r attribute. in-place.

        Args:
            keep_mask (binary int, optional): mask for keep choice of dice. Defaults to 0b00000(reroll every thing).

        """        
        for i, num in enumerate(f"{keep_mask:05b}"):
            if i != '1':
                self.r[num] = self.rng.randint(1, self.NUMBER_OF_SIDES + 1)
    
    def count_numbers(self, dice : np.ndarray = None):
        occurence = np.zeros(self.NUMBER_OF_SIDES, dtype = np.int32)
        if dice is None:
            dice = self.r
        for i in dice:
            occurence[i-1] += 1
        return occurence
    
    def has_one_pair(self):
        occurence = self.count_numbers()
        return np.sum(occurence == 2) == 1
    
    def has_two_pair(self):
        occurence = self.count_numbers()
        return np.sum(occurence == 2) == 2
    

    def has_full_house(self):
        occurence = self.count_numbers()
        return np.any(occurence == 2) and np.any(occurence == 3)
    
    def has_three_of_a_kind(self):
        occurence = self.count_numbers()
        return np.any(occurence == 3)
    
    def has_four_of_a_kind(self):
        occurence = self.count_numbers()
        return np.any(occurence == 4)
    
    def has_small_straight(self):
        seriesness = 0
        for i in range(len(self.r)-1):
            if self.r[i] + 1 == self.r[i+1]:
                seriesness += 1
                if seriesness == 3:
                    return True
        return False
    
    def has_large_straight(self):
        for i in range(len(self.r)-1):
            if self.r[i] + 1 == self.r[i+1]:
                continue
            else:
                return False
        return True
    
    def has_yahtzee(self):
        occurence = self.count_numbers()
        return np.any(occurence == 5)
        
    def count_k(self, k : int) -> int:
        """Count how many k is shown in the roll. k = 1~"""
        occurence = self.count_numbers()
        return occurence[k-1]
    
    
    def get_all_keep_choices_bitmask(self):
        """
        주어진 주사위 결과에서 중복을 고려하여 고유한 keep 선택지를 bitmask (2진수 정수) 형태로 반환한다.
        각 bitmask는 원래 dice 리스트의 인덱스 순서대로 '1'(유지) 또는 '0'(버림)을 나타낸다.
        동일한 값의 주사위에서는, 예를 들어 [1, 1]이 있다면, 1개를 선택하는 경우 항상 첫번째 1을 선택하는 형태로 canonical하게 결정된다.
        """
        # 값 별로 인덱스를 그룹핑 (원래 순서 유지)
        dice = self.r
        groups = defaultdict(list)
        for idx, value in enumerate(dice):
            groups[value].append(idx)
        
        # 그룹 순서는 dice에서 처음 등장하는 순서대로 정렬
        group_keys = sorted(groups.keys(), key=lambda x: groups[x][0]) # [1,1,2,3,4] => [1,2,3,4]
        
        results = []
        n = len(dice)
        
        def backtrack(group_idx, current_mask):
            # 모든 그룹에 대해 처리한 경우 현재 bitmask를 결과에 추가
            if group_idx == len(group_keys):
                results.append("".join(current_mask))
                return
            
            key = group_keys[group_idx]
            indices = groups[key] # key 가 등장했던 모든 index 들
            count = len(indices)
            # 해당 그룹에서는 0부터 count개까지 선택할 수 있음.
            # canonical하게 선택하려면, k개를 선택하는 경우 항상 그룹 내 가장 앞쪽 k개 인덱스를 선택.
            for k in range(count + 1):
                new_mask = current_mask[:]  # 현재 bitmask 복사
                for pos_idx, dice_idx in enumerate(indices):
                    new_mask[dice_idx] = '1' if pos_idx < k else '0'
                backtrack(group_idx + 1, new_mask)
        
        # 초기 bitmask: 모든 주사위에 대해 선택하지 않은 상태('0')로 초기화
        initial_mask = ['0'] * n
        backtrack(0, initial_mask)
        
        for i, item in enumerate(results):
            results[i] = int(item, 2)
        
        return results # e.g) [0b00000, 0b00001, ...]

    def transition_prob(self, keep_mask : int, outcome : np.ndarray):
        nums_to_keep = []
        for i, num in enumerate(f"{keep_mask:05b}"):
            if i == '1':
                nums_to_keep.append(self.r[num])
        for num in nums_to_keep:
            if num not in outcome:
                return 0
        k = Roll.NUMBER_OF_DICE - len(nums_to_keep)
        target = sorted([x for x in outcome if x not in nums_to_keep])
        
            
        return 1/(6**k)*Roll.count_sorted_permutations(target)
    
    
    @staticmethod
    def transition_prob_static(r: np.ndarray,keep_mask : int, outcome : np.ndarray):
        nums_to_keep = []
        for i, num in enumerate(f"{keep_mask:05b}"):
            if i == '1':
                nums_to_keep.append(r[num])
                
        for num in nums_to_keep:
            if num not in outcome:
                return 0
        k = Roll.NUMBER_OF_DICE - len(nums_to_keep)
        return 1/(6**k)
        
    @staticmethod
    def count_numbers_from_dice(dice : np.ndarray):
        occurence = np.zeros(Roll.NUMBER_OF_SIDES, dtype = np.int32)
        for i in dice:
            occurence[i-1] += 1
        return occurence

    @staticmethod
    def choose_by_keepmask(dice : np.ndarray, keep_mask : int):
        nums_to_choose=[]
        for i, num in enumerate(f"{keep_mask:05b}"):
            if i != '1':
                nums_to_choose.append(dice[num])
        nums_to_choose.sort()
        return tuple(nums_to_choose)
    
    @staticmethod
    def count_sorted_permutations(lst):
        count = Counter(lst)
        total = math.factorial(len(lst))
        for freq in count.values():
            total //= math.factorial(freq)
        return total
    
        
    
    @staticmethod
    def get_all_keep_choices_bitmask_from_list(dice : list):
        """
        주어진 주사위 결과에서 중복을 고려하여 고유한 keep 선택지를 bitmask (2진수 정수) 형태로 반환한다.
        각 bitmask는 원래 dice 리스트의 인덱스 순서대로 '1'(유지) 또는 '0'(버림)을 나타낸다.
        동일한 값의 주사위에서는, 예를 들어 [1, 1]이 있다면, 1개를 선택하는 경우 항상 첫번째 1을 선택하는 형태로 canonical하게 결정된다.
        """
        # 값 별로 인덱스를 그룹핑 (원래 순서 유지)
        groups = defaultdict(list)
        for idx, value in enumerate(dice):
            groups[value].append(idx)
        
        # 그룹 순서는 dice에서 처음 등장하는 순서대로 정렬
        group_keys = sorted(groups.keys(), key=lambda x: groups[x][0]) # [1,1,2,3,4] => [1,2,3,4]
        
        results = []
        n = len(dice)
        
        def backtrack(group_idx, current_mask):
            # 모든 그룹에 대해 처리한 경우 현재 bitmask를 결과에 추가
            if group_idx == len(group_keys):
                results.append("".join(current_mask))
                return
            
            key = group_keys[group_idx]
            indices = groups[key] # key 가 등장했던 모든 index 들
            count = len(indices)
            # 해당 그룹에서는 0부터 count개까지 선택할 수 있음.
            # canonical하게 선택하려면, k개를 선택하는 경우 항상 그룹 내 가장 앞쪽 k개 인덱스를 선택.
            for k in range(count + 1):
                new_mask = current_mask[:]  # 현재 bitmask 복사
                for pos_idx, dice_idx in enumerate(indices):
                    new_mask[dice_idx] = '1' if pos_idx < k else '0'
                backtrack(group_idx + 1, new_mask)
        
        # 초기 bitmask: 모든 주사위에 대해 선택하지 않은 상태('0')로 초기화
        initial_mask = ['0'] * n
        backtrack(0, initial_mask)
        
        for i, item in enumerate(results):
            results[i] = int(item, 2)
        
        return results # e.g) [0b00000, 0b00001, ...]


    
    


class ScoreCard:
    CATEGORY_LIST = ['1','2','3','4','5','6','CH','FH','3K','4K','SS','LS','YA']
    LENGTH_OF_CATEGORY = len(CATEGORY_LIST)
    BONUS_THRESHOLD = 63
    
    def __init__(self):
        self.usedC = []
        self.current_scores = np.zeros(self.LENGTH_OF_CATEGORY, dtype = np.int32) # 1D ndarray
    
    @staticmethod
    def print_category_names():
        print(ScoreCard.CATEGORY_LIST)
        
    def get_upscore(self):
        return np.sum(self.current_scores[0:6])
    
    def get_total_score(self):
        return np.sum(self.current_scores)
    
    @staticmethod
    def get_score_for_category(dice : Roll , category : int):
        """_summary_

        Args:
            dice (Roll): Roll object. 
            category (int): 0-12, total 13 categories for yahtzee

        Returns:
            _type_: _description_
        """
        if isinstance(category, str):
            category = ScoreCard.CATEGORY_LIST.index(category) # if category is string, convert it to integer index
        if ScoreCard.is_upper_section(category):
            return dice.count_k(category+1) * (category+1)
        elif ScoreCard.is_lower_section(category):
            if ScoreCard.CATEGORY_LIST[category] == 'CH':
                return np.sum(dice.r)
            elif ScoreCard.CATEGORY_LIST[category] == 'FH':
                return 25 if dice.has_full_house() else 0
            elif ScoreCard.CATEGORY_LIST[category] == '3K':
                return np.sum(dice.r) if dice.has_three_of_a_kind() else 0
            elif ScoreCard.CATEGORY_LIST[category] == '4K':
                return np.sum(dice.r) if dice.has_four_of_a_kind() else 0
            elif ScoreCard.CATEGORY_LIST[category] == 'SS':
                return 30 if dice.has_small_straight() else 0
            elif ScoreCard.CATEGORY_LIST[category] == 'LS':
                return 40 if dice.has_large_straight() else 0
            elif ScoreCard.CATEGORY_LIST[category] == 'YA':
                return 50 if dice.has_yahtzee() else 0
        else:
            logger.critical(f"Unconsidered case occured. category = {category}, dice= {dice.r}, Category = {ScoreCard.CATEGORY_LIST[category]}")
            return 0

    def fill_score(self,dice : Roll,category):
        if isinstance(category, str):
            category = self.CATEGORY_LIST.index(category)
        self.current_scores[category] = self.get_score_for_category(dice,category)
        self.usedC.append(category)
        
    
    @staticmethod
    def is_upper_section(category):
        if isinstance(category,str):
            return str in ['1','2','3','4','5','6']
        else:
            return category in [0,1,2,3,4,5]
    
    @staticmethod
    def is_lower_section(category):
        if isinstance(category, str):
            return str in ['CH','FH','3K','4K','SS','LS','YA']
        else:
            return category in [6,7,8,9,10,11,12]
        
    

class YahtzeeEnvDP(ScoreCard, Roll):

    def __init__(self, reroll = 3, seed=None) -> None:
        self.currentRoll = Roll(seed)
        self.scorecard = ScoreCard()
        self.upscore = self.scorecard.get_upscore() # 0-63
        self.left_reroll = reroll # 0-3
        
    
    def reset(self):
        self.currentRoll = self.currentRoll.reroll()
        self.scorecard = ScoreCard()
        self.upscore = self.scorecard.get_upscore()
        self.reroll = 3
    
            

    @staticmethod
    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load_object(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    

