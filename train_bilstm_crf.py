    
from models import BiLSTM_CRF
import torch.optim as optim
import torch
from tqdm import tqdm

from utils import BiLSTM_CRF_Dataset
from evaluation import evaluate



def test_and_write(mdl, test_data: BiLSTM_CRF_Dataset, outfile: str):
    
    true_tags = []
    pred_tags = []
    
    with torch.no_grad():
        for sentence, tags in tqdm(test_data.iterate(), desc="Testing... ", total=len(test_data.data)):
            targets = [train_data.tag_to_idx[t] for t in tags]

            preds = mdl(sentence)[1]
            
            pred_tags.extend([test_data.idx_to_tag[p] for p in preds])
            true_tags.extend([test_data.idx_to_tag[t] for t in targets])
            
    evaluate(true_tags, pred_tags, verbose=True)
    
        

def train_bilstm_crf(model, train_data: BiLSTM_CRF_Dataset, epochs=1):
    """ wrapper function to train a bilstm-crf """
    
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters())
    print("#################################################")
    print("# Starting training ...")
    print("#################################################")
    
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in tqdm(train_data.iterate(), desc=f"Epoch {epoch}: ", total=len(train_data.data)):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            targets = torch.tensor([train_data.tag_to_idx[t] for t in tags], dtype=torch.long)

            loss = model.neg_log_likelihood(sentence, targets)

            loss.backward()
            optimizer.step()
            
    print("#################################################")
    
    return model



    
if __name__=="__main__":
    
    # output files
    UNTRAINED_OUTFILE = "generated/untrained.out"
    TRAINED_OUTFILE = "generated/trained.out"
    
    train_data = BiLSTM_CRF_Dataset("train")
    test_data = BiLSTM_CRF_Dataset("test")
    
    EMBEDDING_DIM = 200
    HIDDEN_DIM    = 86
    
    print("#################################################")
    print("# Building model ...  ")
    print("#################################################")
    model = BiLSTM_CRF(train_data.tag_to_idx, train_data.word_to_idx, EMBEDDING_DIM, HIDDEN_DIM, word2vec_embeds=True)
    model = train_bilstm_crf(model, train_data, epochs=5)
    
    # Check predictions after training
    test_and_write(model, test_data, TRAINED_OUTFILE)