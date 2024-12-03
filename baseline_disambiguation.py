

from utils import load_meddra_id2name, DisambiguationDataset
from evaluation import evaluation_report   
import editdistance


def find_matches(term_list, vocab: dict[int, str]):
    """ for each word in the word_list, return the closest match (ID) of vocab using Levenshtein distance """
    matches = []
    for term in term_list:
        closest, closestDist = None, 1e5
        for id, word in vocab.items():
            dist = editdistance.eval(term, word)
            if dist < closestDist:
                closest = id
                closestDist = dist
                
        matches.append(closest)
    
    return matches


if __name__=="__main__":
    
    print("Starting training")
    print("-----------------")
    meddra = load_meddra_id2name()
    print(" * Loaded MEDDRA")
    
    # here, we need to generate embeddings for each sentence in the meddra database
    
    
    dd = DisambiguationDataset()
    print(" * Loaded dataset")
    print("-----------------")
    
    
    true_reactions = dict()
    pred_reactions = dict()
    
    STOP, curr = 30, 0
    
    for name, data in dd.iterate():
        
        if curr > STOP:
            break
        
        print(curr, name)
        # for each reaction, add the meddra_pt_id in int form to the ground truth
        true_reactions[name] = [int(r) for r in data["reactions"] if r is not None]
        # get closest ids for each mention
        pred_reactions[name] = find_matches(data["mentions"], meddra)
        
        curr += 1
            
    print("-----------------")
    print(" * calculating f1s ")
    
        
    avg_f1 = 0
    avg_precision = 0
    avg_recall = 0
    for drug in true_reactions.keys():
        report = evaluation_report(pred_reactions[drug], true_reactions[drug])
        avg_f1 += report[0]
        avg_precision += report[1]
        avg_recall += report[2]
        
    avg_f1 /= len(true_reactions.keys())
    avg_precision /= len(true_reactions.keys())
    avg_recall /= len(true_reactions.keys())
        
    print(f"F1: {avg_f1:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
        
        
    
    