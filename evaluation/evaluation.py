
from typing import Iterable


def evaluation_report(output: Iterable[str], known_set: Iterable[str]) -> tuple[float, float, float]:
    """ generates f1 score for a single example"""
    
    overlap = set(output).intersection(set(known_set))
    
    tp = len(overlap)                   # true positive is the union of the output and known_set
    fp = len(output) - len(overlap)     # false positive is every guess in output that is not in known_set, or output - overlap
    fn = len(known_set) - len(overlap)  # false negative is every guess in known_set that is not in output, or known_set - overlap
    
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    
    f1 = 2*precision*recall / (precision+recall)
    
    return f1, precision, recall