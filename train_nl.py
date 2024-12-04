from models import DisambigModel, DisambigRuleModel
from utils import DisambiguationDataset
from evaluation import evaluation_report
from tqdm import tqdm

if __name__=="__main__":
    
    dd = DisambiguationDataset()
    model = DisambigRuleModel()

    true_reactions = dict()
    pred_reactions = dict()
    
    print("-----------------")
    print(" * generating results")
    for name, data in tqdm(dd.iterate(), total=len(dd.all_drugs)):
        # for each reaction, add the meddra_pt_id in int form to the ground truth
        true_reactions[name] = [int(r) for r in data["reactions"] if r is not None]
        
        # get closest ids for each mention
        pred_reactions[name] = model.predict_ids(data["mentions"])
        
    
    print("-----------------")
    print(" * calculating f1s ")
    
    avg_f1 = 0.0
    avg_precision = 0.0
    avg_recall = 0.0
    for drug in true_reactions.keys():
        try:
            report = evaluation_report(pred_reactions[drug], true_reactions[drug])
        except Exception as e:
            print(e)
            print("Error for:", drug)
            # print(dd.all_drugs[drug]["mentions"])
        avg_f1 += report[0]
        avg_precision += report[1]
        avg_recall += report[2]
        
    avg_f1 /= len(true_reactions.keys())
    avg_precision /= len(true_reactions.keys())
    avg_recall /= len(true_reactions.keys())
        
    print(f"F1: {avg_f1:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
        
        
    
    
