


def evaluate_f1(output, known_set) -> float:
    """ generates f1 score for a single example"""
    
    overlap = set(output).intersection(set(known_set))
    
    tp = len(overlap)                   # true positive is the union of the output and known_set
    fp = len(output) - len(overlap)     # false positive is every guess in output that is not in known_set, or output - overlap
    fn = len(known_set) - len(overlap)  # false negative is every guess in known_set that is not in output, or known_set - overlap
    
    score = (2 * tp) / (2*tp + fp + fn)
    
    return score