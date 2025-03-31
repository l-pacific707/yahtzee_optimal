from YahtzeeEnvDP import Roll,ScoreCard,YahtzeeEnvDP
from reachable import load_reachable_state, _all_subsets_bitmask
from collections import defaultdict
import logging
import numpy as np
import pickle
import logger_setup
from collections import Counter
import math
import time

logger = logging.getLogger(__name__)

'''
R(n,S) is boolean value that indicates whether it is possible to get n score in upper part using category in S.
- R(0,S) is True for all S 
- R(n, emptyset) = False for any n >=1 
'''

# state definition
'''
S = (C,m,f)
C = categories that was already marked for score
m = upperscore total
f = flag that Yahtzee was filled or not (for Yahtzee bonus, we don't care here)

r : dice roll result
rk : keep dice choice. e.g) [1,2]  in r = [1,2,2,3,3]
c : category that is currently chosen for scoring

E(S,r,n) = E((C,m,f),r,n) : potential of state S with n rerolls remaining and r dice results.
'''


def find_valid_Cm_pair(filename = "valid_Cm_pair.pkl"):
    try:
        with open(filename, 'rb') as f:
            valid_Cm_pair = pickle.load(f)
        return valid_Cm_pair
    except FileNotFoundError:
        subCs = _all_subsets_bitmask(range(1, len(ScoreCard.CATEGORY_LIST)+1), reverse=True) # all subset of category (2**13 = 8192개), {전체, 전체-1, 전체-2,...,1}
        # subset of [1,2,3,...,13]

        '''
        By ensuring the computation done for any state with more categories marked, we have all information to calculate E(S)
        '''
        logger.debug(f"type of elements in subCs: {type(subCs[0])}")
        logger.debug(f"first three elements in subCs: {subCs[:3]}")
        R = load_reachable_state("reachable_state.pkl")
        reachables = {}
        valid_Cm_pair = []
        upsubset_to_index = {}
        upsubset = _all_subsets_bitmask(range(1, 7))
        for i, u in enumerate(upsubset):
            upsubset_to_index[u] = i

        for m in range(ScoreCard.BONUS_THRESHOLD + 1):
            reachables[m] = {}
            for C in subCs:
                uppercat_part = [x for x in C if x <= 6]
                uppercat_part.sort()
                idx = upsubset_to_index[tuple(uppercat_part)]
                if R[m, idx]:
                    reachables[m][C] = True
                    valid_Cm_pair.append((C, m))
                else:
                    reachables[m][C] = False
        with open(filename, 'wb') as f:
            pickle.dump(valid_Cm_pair, f)
        with open("reachables_truthtable_m_C.pkl", 'wb') as f:
            pickle.dump(reachables, f)
        return valid_Cm_pair

def update_expectation_for_state(expectation : defaultdict, cmpair : tuple,f:bool,r : np.ndarray, rk : int , n : int,
                              processed = None, total = None, start_time = None):
    """

    Args:
        expectation (nested dict): E[(C,m,f)][r][n]
        cmpair (tuple): (C,m)
        f (bool) : yahtzee bonus flag
        r (ndarray or list ): dice result
        rk (integer): it is keep choice mask binary.
        n (int): remaining reroll
    """
    C,m = cmpair
    
    # debug purpose logging
    if processed is not None and total is not None and start_time is not None:
        if processed % 1000 == 0 or processed == total:
            elapsed = time.time() - start_time
            percent = processed / total
            if percent > 0:
                estimated_total_time = elapsed / percent
                eta = estimated_total_time - elapsed
                logger.info(
                    f"[{processed}/{total}] "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"ETA: {eta:.1f}s | "
                    f"Progress: {percent*100:.2f}%"
                )
    
    if len(C) == len(ScoreCard.CATEGORY_LIST):
        logger.debug(f"full categories are used. Game end. current bonus value : {m}")
        if m >= ScoreCard.BONUS_THRESHOLD:
            expectation[(C,m,f)][tuple(r)][n] = 35
        else:
            expectation[(C,m,f)][tuple(r)][n] = 0
        return -1
    elif n == 0:
        logger.debug(f"End of the turn.  so n = 0. current category used: {C}, current roll result : {r}")
        clist = []
        for i in range(1, len(ScoreCard.CATEGORY_LIST)+1):
            if i not in C:
                clist.append(i)
        for i, c in enumerate(clist):
            newC = tuple(sorted(C + (c,)))
            clist[i] = ScoreCard.get_score_for_category(Roll(r),c) + expectation[(newC,m,f)][r][n]
        expectation[(C,m,f)][tuple(r)][n] = max(clist)
        return C[np.argmax(clist)]
    elif rk == 2**Roll.NUMBER_OF_DICE - 1:
        # keep everything
        expectation[(C,m,f)][tuple(r)][n] = expectation[(C,m,f)][tuple(r)][0]
    elif rk is not None:
        val = 0
        for outcome in Roll.ALL_DICE_STATES:
            val += Roll.transition_prob_static(r,rk,outcome)*expectation[(C,m,f)][outcome][n-1]
        expectation[(C,m,f)][Roll.choose_by_keepmask(r,rk)][n] = val 
    
    elif rk is None:
        temp = expectation[(C,m,f)][tuple(r)][n]
        maxchoice = None
        for keepchoice in Roll.get_all_keep_choices_bitmask_from_list(r):
            newval = expectation[(C,m,f)][Roll.choose_by_keepmask(r,keepchoice)][n] 
            if temp  <= newval:
                temp = newval
                maxchoice = keepchoice
        expectation[(C,m,f)][tuple(r)][n] = temp
        return maxchoice # e.g) [1,2]
    
    elif np.sum(r) == 0:
        # beginning of the turn
        val = 0
        for result in Roll.ALL_DICE_STATES:
            val += expectation[(C,m,f)][result][2]*Roll.transition_prob_static(r,0,result)
        expectation[(C,m,f)][tuple(r)][n] = val
        logger.debug(f"expectation for entry point was done. (C,m,f) = {C,m,f}, r= {r}, n={n}")

def update_expectation_end_of_turn(expectation, C, m, f, r):
    """
    n 이 0 일때만 호출됨.
    E((C,m,f), r, n=0).
    We choose the best category from the ones not in C,
    referencing the next-state's E-value.
    """
    if len(C) == len(ScoreCard.CATEGORY_LIST):
        # All categories used → final
        if m >= ScoreCard.BONUS_THRESHOLD:
            expectation[(C,m,f)][tuple([0 for _ in range(Roll.NUMBER_OF_DICE)])][3] = 35
        else:
            expectation[(C,m,f)][tuple([0 for _ in range(Roll.NUMBER_OF_DICE)])][3] = 0
        return

    # Otherwise, pick best category
    candidates_val = []
    candidates_cat = []
    cats_not_used = [cat for cat in range(1, len(ScoreCard.CATEGORY_LIST)+1) if cat not in C]
    for cat in cats_not_used:
        C_next = tuple(sorted(C + (cat,)))
        # Score from the chosen category on the current roll
        points_here = ScoreCard.get_score_for_category(Roll(r), cat-1) # index is needed here. so cat -1
        
        # if points_here ==50 and ScoreCard.CATEGORY_LIST[cat-1] == 'YA':
        #     newf = True
        # else:
        #     newf = f
        newf = f
        
        # Then add the expected value E(S') where S' = (C_next, updated_m, f)
        # But you have to check if cat is an upper category
        new_m = m + points_here if ScoreCard.is_upper_section(cat) else m
        # E(S') means "start of next turn" → we typically store that in something like E(S') = ...
        # For now, let's assume we store *turn-level* expectation as well
        # or we re-use E[(C_next, new_m, f)][anything][some_n].
        # A simple approach is to store the "start-of-turn" value in E_start[(C,m,f)].
        # We'll just do something simplistic here:
        # We'll define a helper function or we keep E_start in a separate dict.

        # Example: if you store "start-of-turn" in E_start:
        next_val = expectation[(C_next, new_m, newf)][tuple([0 for _ in range(Roll.NUMBER_OF_DICE)])][3]  # ensure it's computed
        try:
            candidates_val.append(points_here + next_val)
        except TypeError:
            logger.debug(f"type of next_val: {type(next_val)}")
            logger.debug(f"value of next_val: {next_val}, nextC = {C_next}, nextm = {new_m}, nextf = {newf}")
            logger.debug(f"type of points_here: {type(points_here)}")
            logger.debug(f"current category being considered: {cat}")
        candidates_cat.append(cat)

    best_val = max(candidates_val) if candidates_val else 0
    best_cat = candidates_cat[np.argmax(candidates_val)] if candidates_val else None
    expectation[(C,m,f)][tuple(r)][0] = best_val
    return best_cat

def update_expectation_reroll_stage(expectation, C, m, f, r, n):

    # We'll do it inline for clarity:
    best_val = -99
    subrolls = Roll.get_all_keep_choices_bitmask_from_list(r)  # all possible r' subsets
    # type of subroll : integer
    best_subroll = None
    for keepmask in subrolls:
        # compute "reroll expectation"
        val = 0
        for outcome in Roll.ALL_DICE_STATES:
            p = Roll.transition_prob_static(r, keepmask, outcome)
            val_outcome = expectation[(C,m,f)][outcome][n-1]
            val += p * val_outcome

            # 2) track the best among subrolls
            if val > best_val:
                best_val = val
                best_subroll = keepmask

    # store result
    expectation[(C,m,f)][tuple(r)][n] = best_val
    return best_subroll

def initialize_expectation(expectation, m, f):
    C = tuple(range(1, len(ScoreCard.CATEGORY_LIST) + 1))
    for r in Roll.ALL_DICE_STATES:
        # no matter what r is, C is full. So bonus point is the only reward available.
        if m >= ScoreCard.BONUS_THRESHOLD:
            expectation[(C,m,f)][r][0] = 35
        else:
            expectation[(C,m,f)][r][0] = 0
    

def fill_dp_table(processed = None, total = None, start_time = None):
    # Our global DP structure: E(S,r,n).
    # Keys: ((C,m,f), r, n). We'll store as: expectation[(C,m,f)][r][n]
    expectation = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    # We also store a "start-of-turn" dictionary: E_start[S] = E(S). 
    # i.e. if you're starting a turn in state S before rolling, what's the expected value?

    valid_Cm_pair = find_valid_Cm_pair()  # or load from file
    # Sort from largest subset to smallest
    valid_Cm_pair_sorted = sorted(valid_Cm_pair, key=lambda x: len(x[0]), reverse=True)

    # We'll assume f only has two possibilities: [False, True]
    # If you always do f=False for now, that's fine. 
    # If you do handle it, also loop f in [False, True].
    
    for (C, m) in valid_Cm_pair_sorted:
            f = False
            S = (C, m, f)

            #
            # (A) Compute E_start[S] = sum_{all first-roll r} P(r)* E(S,r,2).
            # But first we must fill E(S,r,n) for n=2,1,0. 
            
                # debug purpose logging
            if processed is not None and total is not None and start_time is not None:
                if processed % 1000 == 0 or processed == total:
                    elapsed = time.time() - start_time
                    percent = processed / total
                    if percent > 0:
                        estimated_total_time = elapsed / percent
                        eta = estimated_total_time - elapsed
                        logger.info(
                            f"[{processed}/{total}] "
                            f"Elapsed: {elapsed:.1f}s | "
                            f"ETA: {eta:.1f}s | "
                            f"Progress: {percent*100:.2f}%"
                        )

            # Step 1: n=0 for all r
            # This references next states with bigger subsets. 
            # Because we've sorted from big to small, those bigger subsets E_start[] is known.
            for r in Roll.ALL_DICE_STATES[1:]:
                update_expectation_end_of_turn(expectation, C, m, f, r)

            # Step 2: n=1 for all r
            for r in Roll.ALL_DICE_STATES[1:]:
                # E(S, r, 1): pick dice to keep and sum over outcomes => E(S,outcome,0)
                update_expectation_reroll_stage(expectation, C, m, f, r, n=1)

            # Step 3: n=2 for all r
            for r in Roll.ALL_DICE_STATES[1:]:
                update_expectation_reroll_stage(expectation, C, m, f, r, n=2)

            # Step 4: now E_start[S] = sum_{r} P(r) * E(S, r, 2)
            # "P(r)" is just the probability of rolling r from scratch:  (1/6^5)* (count_of_that_combination).
            # We'll assume Roll.roll_probability(r) is provided or we can compute from combinations.
            val = 0.0
            for r in Roll.ALL_DICE_STATES[1:]:
                p_r = Roll.transition_prob_static([0,0,0,0,0],0,r)
                val += p_r * expectation[S][r][2]
            expectation[S][tuple([0 for _ in range(Roll.NUMBER_OF_DICE)])][3] = val
            
            processed += 1

    # done filling everything
    return expectation

if __name__ == "__main__":
    logger_setup  # side effect for logging.basicConfig, if needed

    try:
        with open("expectation.pkl", 'rb') as f:
            expectation = pickle.load(f)
        logger.info("Loaded existing DP table from expectation.pkl")
    except FileNotFoundError:
        logger.info("No existing DP file found. Building from scratch...")
        start_time = time.time()

        # Fill the DP table in the correct order
        expectation = fill_dp_table(processed=0, total = 2**19, start_time=start_time)

        with open("expectation.pkl", 'wb') as f:
            pickle.dump(expectation, f)
        logger.info(f"Done! Took {time.time()-start_time:.2f} seconds.")
        logger.info(f"Saved in 'expectation.pkl' file.")
