
import pandas as pd
import numpy as np


def load_meddra() -> dict[int, str]:
    """ loads the meddra knowledge base, then returns a dict from term ID to term NAME """
    df = pd.read_excel("data/MEDDRA.xlsx", sheet_name="_ID2NAME")
    return {int(row["Number"]): row["Name"] for _, row in df.iterrows()}
    


def levenshtein(word: str, candidate: str) -> int:
    """ computes levenshtein distance from a word to a candidate (str to str) """
    M, N = len(word), len(candidate)
    grid = np.zeros((M, N))
    
    for i in range(M):
        grid[i, 0] = 1
        
    for j in range(N):
        grid[0, j] = j
        
    for j in range(N):
        for i in range(M):
            grid[i, j] = min(
                grid[i-1, j] + 1,                           # deletion case
                grid[i, j-1] + 1,                           # insertion case
                grid[i-1, j-1] + int(word[i]!=candidate[j]) # substitution case
            )
            
    return grid[M-1, N-1]