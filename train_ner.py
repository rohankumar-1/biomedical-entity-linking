    
from models import BiLSTM_CRF, BaseLSTM
import torch.optim as optim
import torch
from tqdm import tqdm
import os

from utils import IOBDataset, E2EDataset
from evaluation import evaluate



def test_and_write(mdl, test_data: E2EDataset, outdir: str):
    
    true_tags = []
    pred_tags = []
    
    with torch.no_grad():
        for drug, text, tagset, _ in tqdm(test_data.iterate(), desc="Testing... ", total=len(test_data.sentset)):
            with open(os.path.join(outdir, f"{drug}_tagged.out"), "w") as fp:
                for sentence, tags in zip(text, tagset):
                    
                    preds = mdl(sentence)[1]
                    
                    pred_tags.extend([test_data.idx_to_tag[p] for p in preds])
                    true_tags.extend(tags)
                    
                    for word, tag in zip(sentence, preds):
                        print(f"{word} {test_data.idx_to_tag[tag]}", file=fp)
                    print(" ", file=fp)
                
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
    
    train_data = IOBDataset("train")
    test_data = E2EDataset()
    
    EMBEDDING_DIM = 200
    HIDDEN_DIM    = 128
    
    print("Training model...")
    ######################################################
    # set model to train below
    model = train_bilstm_crf(train_data, 
                             epochs=20, 
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
    
    print("Testing model and generating results...")
    
    # Check predictions after training, write to output
    TRAINED_OUTDIR = "generated/tagged/"
    
    test_and_write(model, test_data, TRAINED_OUTDIR)