    
from models import BiLSTM_CRF, BaseLSTM
import torch.optim as optim
import torch
from tqdm import tqdm

from utils import IOBDataset
from evaluation import evaluate



def test_and_write(mdl, test_data: IOBDataset, outfile: str):
    
    true_tags = []
    pred_tags = []
    
    with torch.no_grad():
        for sentence, tags in tqdm(test_data.iterate(), desc="Testing... ", total=len(test_data.data)):
            targets = [train_data.tag_to_idx[t] for t in tags]

            preds = mdl(sentence)
            
            pred_tags.extend([train_data.idx_to_tag[torch.argmax(p).item()] for p in preds])
            true_tags.extend([train_data.idx_to_tag[t] for t in targets])
            
    evaluate(true_tags, pred_tags, verbose=True)
    

def train_lstm(train_data: IOBDataset, epochs=1, embedding_dim=128, hidden_dim=100):
    
    model = BaseLSTM(train_data.tag_to_idx,
                     train_data.word_to_idx,
                     embedding_dim=embedding_dim,
                     hidden_dim=hidden_dim,
                    )
    
    optimizer = optim.AdamW(model.parameters())
    
    for e in range(epochs):
        for sentence, tags in tqdm(train_data.iterate(), desc=f"Epoch {e}: ", total=len(train_data.data)):
            model.zero_grad()
            targets = torch.tensor([train_data.tag_to_idx[t] for t in tags], dtype=torch.long)
    
            out = model(sentence)
            loss = torch.nn.functional.cross_entropy(out, targets)
            
            loss.backward()
            optimizer.step()
        
    return model     


def train_bilstm_crf(train_data: IOBDataset, epochs=1, embedding_dim=200, hidden_dim=100, word2vec_embeds=True):
    """ wrapper function to train a bilstm-crf """

    model = BiLSTM_CRF(train_data.tag_to_idx, 
                       train_data.word_to_idx, 
                       embedding_dim=embedding_dim, 
                       hidden_dim=hidden_dim, 
                       word2vec_embeds=word2vec_embeds
                       )
    
    optimizer = optim.AdamW(model.parameters())
    
    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in tqdm(train_data.iterate(), desc=f"Epoch {epoch}: ", total=len(train_data.data)):
            
            model.zero_grad()
            targets = torch.tensor([train_data.tag_to_idx[t] for t in tags], dtype=torch.long)

            loss = model.neg_log_likelihood(sentence, targets)

            loss.backward()
            optimizer.step()
    
    return model

    
if __name__=="__main__":
    
    # output files
    UNTRAINED_OUTFILE = "generated/untrained.out"
    TRAINED_OUTFILE = "generated/trained.out"
    
    train_data = IOBDataset("train")
    test_data = IOBDataset("test")
    
    EMBEDDING_DIM = 200
    HIDDEN_DIM    = 128
    
    print("Training model...")
    ######################################################
    # set model to train below
    model = train_bilstm_crf(train_data, 
                             epochs=5, 
                             embedding_dim=EMBEDDING_DIM, 
                             hidden_dim=HIDDEN_DIM,
                             word2vec_embeds=True
                             )
    
    # model = train_lstm(train_data,
    #                    epochs=5,
    #                    embedding_dim=EMBEDDING_DIM,
    #                    hidden_dim = HIDDEN_DIM,
    #                     )
    ######################################################
    
    print("Testing model...")
    # Check predictions after training
    test_and_write(model, test_data, TRAINED_OUTFILE)