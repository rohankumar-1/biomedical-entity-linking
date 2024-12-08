
import editdistance
from utils.utils import load_meddra_id2name


class LevenshteinModel():
    
    def __init__(self):
        self.meddra: dict[int, str] = load_meddra_id2name()
    
    def find_matches(self, term_list):
        """ for each word in the word_list, return the closest match (ID) of vocab using Levenshtein distance """
        matches = []
        for term in term_list:
            closest, closestDist = None, 1e5
            for id, word in self.meddra.items():
                dist = editdistance.eval(term, word)
                if dist < closestDist:
                    closest = id
                    closestDist = dist
                    
            matches.append(closest)
        
        return matches
    