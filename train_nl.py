from models import DisambigModel, DisambigRuleModel, LevenshteinModel
from utils import DisambiguationDataset
from evaluation import evaluation_report
from tqdm import tqdm

def eval_results(true, pred):
    """ given dict of true and dict of pred reactions (keys are drugs), return avg F1, precision, recall """
    avg_f1, avg_precision, avg_recall = 0.0, 0.0, 0.0
    
    N = len(true.keys())
    for drug in true.keys():
        report = evaluation_report(pred[drug], true[drug]) # gets f1, precision, recall for this drug
        
        avg_f1 += report[0]
        avg_precision += report[1]
        avg_recall += report[2]
    
    # divide by N (# drugs) to finalize average
    return avg_f1/N, avg_precision/N, avg_recall/N


def run_leven(data: DisambiguationDataset, stop=10):
    """ build model that uses Levenshtein edit distance between entity term and each term in Meddra, returning ID of smallest distance """
    model = LevenshteinModel()
    
    curr = 0
    true_reactions = dict()
    pred_reactions = dict()
    for n, d in tqdm(data.iterate(), total=stop):
        if curr >= stop:
            break
        
        # for each reaction, add the meddra_pt_id in int form to the ground truth
        true_reactions[n] = [int(r) for r in d["reactions"] if r is not None]
        # get closest ids for each mention
        pred_reactions[n] = model.find_matches(d["mentions"])
        curr += 1
        
    return eval_results(true_reactions, pred_reactions)
 
 
def run_disambig(data: DisambiguationDataset, rules=True):
    """ build model using cosine similarity of embeddings based on BioSentVec embeddings between entity term and MedDRA entries """
    model = DisambigRuleModel() if rules else DisambigModel()
    
    true_reactions = dict()
    pred_reactions = dict()
    
    for name, data in tqdm(data.iterate(), total=len(dd.all_drugs)):
        # for each reaction, add the meddra_pt_id in int form to the ground truth
        true_reactions[name] = [int(r) for r in data["reactions"] if r is not None]
        
        # get closest ids for each mention
        pred_reactions[name] = model.predict_ids(data["mentions"])
        
    return eval_results(true_reactions, pred_reactions)




if __name__=="__main__":
    
    dd = DisambiguationDataset()

    
    print("Generating results...")
    # avg_f1, avg_prec, avg_recall = run_disambig(dd, rules=True)
    avg_f1, avg_prec, avg_recall = run_leven(dd, stop=10)
    
    print(f"F1: {avg_f1:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    
    
        
        
        
    
    
