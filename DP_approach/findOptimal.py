from YahtzeeEnvDP import Roll,ScoreCard,YahtzeeEnvDP
from reachable import load_reachable_state, _all_subsets_bitmask
from itertools import combinations_with_replacement
from itertools import combinations
from itertools import product
from collections import defaultdict
import logging
import numpy as np
import pickle

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
scorecard = ScoreCard()
C = scorecard.usedC
m = scorecard.get_upscore()