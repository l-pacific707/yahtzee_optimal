import pickle
import numpy as np
from YahtzeeEnvDP import ScoreCard, Roll


def find_unreachable_state(load_file_path = None):
    """

    Args:
        load_file_path (string,  optional): load file name(path). Defaults to None.

    Returns:
        R ( reachable state, np.ndarray): shape(64,64), dtype = np.bool
        
    Additional description:
        R(n,S) is boolean value that indicates whether it is possible to get n score in upper part using category in S.
        - R(0,S) is True for all S 
        - R(n, emptyset) = False for any n >=1 
    """
    if load_file_path is not None:
        try:
            with open(load_file_path, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            print(f"File not found: {load_file_path}")
            return None
    else:
        S = [x for x in range(1, Roll.NUMBER_OF_SIDES+1)] # [1,2,3,4,5,6]
        # R is initialized with False value
        R = np.zeros(( ScoreCard.BONUS_THRESHOLD + 1,2**Roll.NUMBER_OF_SIDES), dtype= np.bool) # 0<=n<=63, 2**6 : # of subsets of {1,2,3,4,5,6} (categories to be filled)
        
        subS = _all_subsets_bitmask(S) # list of tuples [(), (1), (2), ...]
        print(f"number of subsets: {len(subS)}")
        subsettoindex = {}
        for i, ss in enumerate(subS):
            subsettoindex[ss] = i
        
        #initialize
        for j, _ in enumerate(subS):
            R[0,j] = True # first row must be True note R[0,0] = True
        
        # find R recursively
        for n in range(1,R.shape[0]): # for n>=1
            for S_idx in range(1, len(subS)): #except empty set
                # essentially we look for R[row>=1:col>=1]
                S= subS[S_idx]
                truth = False
                for x in S:
                    S_without_x = [item for item in S if item != x]
                    idx_S_without_x = subsettoindex[tuple(S_without_x)]
                    
                    for k in range(Roll.NUMBER_OF_DICE+1):
                        if n-k*x >= 0:
                            truth = truth or R[n-k*x, idx_S_without_x]
                            if S_idx == 63:
                                print(f"{subS[S_idx]} is {truth}")
                # after loop over x and k
                R[n,S_idx] = truth
        
        print("first five columns of R")
        print(R[:10,:10])
        
        with open("reachable_state.pkl", 'wb') as f:
            pickle.dump(R, f)
        return R
                
def load_reachable_state(pkl_path="reachable_state.pkl"):
    with open(pkl_path, 'rb') as f:
        R = pickle.load(f)
    return R
    

def _all_subsets_bitmask(s, reverse = False):
    s = list(s)
    n = len(s)
    subsets = []
    for i in range(1 << n):  # 0부터 2^n - 1까지
        subset = [s[j] for j in range(n) if (i >> j) & 1]
        subsets.append(tuple(subset))
    subsets.sort(key=len,reverse=reverse)
    return subsets

def _save_reachable_state_as_markdown(pkl_path="reachable_state.pkl", output_path="reachable_state.md"):
    # Load reachable state
    with open(pkl_path, 'rb') as f:
        R = pickle.load(f)

    # Prepare subset labels
    S = [x for x in range(1, Roll.NUMBER_OF_SIDES + 1)]  # [1,2,3,4,5,6]
    subsets = _all_subsets_bitmask(S)
    subset_headers = [f"{i}: {set(s)}" for i, s in enumerate(subsets)]

    # Start building markdown content
    lines = []

    # Header row
    header_row = "| n \\ subset | " + " | ".join(subset_headers) + " |"
    lines.append(header_row)

    # Separator row
    separator_row = "|" + "---|" * (len(subset_headers) + 1)
    lines.append(separator_row)

    # Each data row
    for n in range(R.shape[0]):  # n from 0 to 63
        row = [f"{n}"]
        for s in range(R.shape[1]):
            val = "✅" if R[n, s] else "❌"
            row.append(val)
        line = "| " + " | ".join(row) + " |"
        lines.append(line)

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown table saved to: {output_path}")

if __name__ == "__main__":
    R= find_unreachable_state()
    print(f"number of reachable state: {np.sum(R)}")
    print(f"number of unreachable state: {np.sum(~R)}")
    print(f"total state: {R.shape[0]*R.shape[1]}")
    _save_reachable_state_as_markdown("reachable_state.pkl", "reachable_state.md")
    
        