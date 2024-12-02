
import pandas as pd
import numpy as np

START_TOKEN = "<START>"
STOP_TOKEN = "<STOP>"
UNK_TOKEN = "<UNK>"


def load_meddra() -> dict[str, str]:
    """ loads the meddra knowledge base, then returns a dict from term ID to term NAME """
    df = pd.read_excel("data/MEDDRA.xlsx", sheet_name="_ID2NAME")
    return {row["Number"]: row["Name"].lower() for _, row in df.iterrows()}
    
    
def load_meddra_id2name() -> dict[str, str]:
    """ loads the meddra knowledge base, then returns a dict from term ID to term NAME """
    df = pd.read_excel("data/MEDDRA.xlsx", sheet_name="_ID2NAME")
    return {int(row["Number"]): row["Name"].lower() for _, row in df.iterrows()}
    

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


def iob_to_entities(tokens, predictions):
    """ convert IOB output to entities given tokens and prediction """
    entities = []
    current_entity = []
    current_label = None

    for token, label in zip(tokens, predictions):
        if label.startswith('B-'):  # Beginning of a new entity
            if current_entity:  # Save the previous entity if it exists
                entities.append((' '.join(current_entity), current_label))
                current_entity = []
            current_entity.append(token["word"])
            current_label = label[2:]  # Get the entity type without 'B-'
        elif label.startswith('I-') and current_entity:  # Inside an existing entity
            current_entity.append(token["word"])
        else:  # Outside an entity
            if current_entity:  # Save the previous entity if it exists
                entities.append((' '.join(current_entity), current_label))
                current_entity = []
                current_label = None

    # Capture any remaining entity
    if current_entity:
        entities.append((' '.join(current_entity), current_label))

    return entities


def write_results(outfile, sentences, true_tags, pred_tags):
    
    with open(outfile, "r") as f:
        for sen, true, pred in zip(sentences, true_tags, pred_tags):
            print("this", file=f)
        
    return 